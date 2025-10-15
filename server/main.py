from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware


def get_allowed_origins() -> list[str]:
    env_val = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return [o.strip() for o in env_val.split(",") if o.strip()]


# Load env file for local dev (.env at project root)
env_path = Path(__file__).resolve().parents[1] / ".env"
print(f"[startup] Loading .env from: {env_path}")
print(f"[startup] .env file exists: {env_path.exists()}")
load_dotenv(dotenv_path=env_path)
print(f"[startup] DEEPGRAM_API_KEY after load_dotenv: {bool(os.getenv('DEEPGRAM_API_KEY'))}")

app = FastAPI(title="STT Proxy", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Phase 2.5: Real-time thought tracker for creating discrete thoughts per speaker turn
class RealTimeThoughtTracker:
    """Tracks thoughts in real-time, creating discrete thoughts for each speaker turn."""

    def __init__(self):
        self.current_thought: dict[str, Any] | None = None
        self.last_speaker: int | None = None

    def add_segment(self, speaker: int, text: str, start_ms: int, end_ms: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """
        Add a segment and return (finalized_previous_thought, current_thought).
        Merges with existing thought if same speaker, creates new if speaker changed.
        Returns (None, current_thought) if same speaker continuing.
        Returns (finalized_thought, new_thought) if speaker changed.
        """
        import uuid

        finalized_thought = None

        # Check if this is the same speaker continuing
        if speaker == self.last_speaker and self.current_thought is not None:
            # Same speaker - append to existing thought
            self.current_thought["text"] += " " + text
            self.current_thought["end_ms"] = end_ms
            self.current_thought["sequence"] += 1
            # Keep as non-final while speaker continues
            self.current_thought["is_final"] = False

            print(f"[ThoughtTracker] Merged to Speaker {speaker}: '{text}' (seq {self.current_thought['sequence']})", flush=True)

        else:
            # Speaker changed - finalize previous and start new
            if self.current_thought is not None:
                self.current_thought["is_final"] = True
                finalized_thought = self.current_thought.copy()
                print(f"[ThoughtTracker] Finalized thought for Speaker {self.last_speaker}", flush=True)

            # Create new thought
            self.current_thought = {
                "id": str(uuid.uuid4()),
                "speaker": f"Speaker {speaker}",
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "sequence": 0,
                "is_final": False
            }

            print(f"[ThoughtTracker] New thought for Speaker {speaker}", flush=True)

        self.last_speaker = speaker

        return (finalized_thought, self.current_thought.copy())


# Phase 2.5: Speaker smoothing to reduce flickering
class SpeakerSmoother:
    """Smooths speaker labels over time to reduce flickering/incorrect switches."""

    # Configuration
    MIN_SPEAKER_DURATION_MS = 800    # Minimum duration to trust a speaker change
    HISTORY_WINDOW_MS = 3000         # Look back window for context
    SPEAKER_CHANGE_THRESHOLD = 0.6   # Confidence threshold for speaker change

    def __init__(self):
        self.word_history: list[dict[str, Any]] = []  # Recent words with timestamps and speakers
        self.current_speaker: int | None = None
        self.current_speaker_start: float = 0.0

    def smooth_speaker(self, word_obj: dict[str, Any]) -> int | None:
        """
        Apply temporal smoothing to speaker labels.

        Args:
            word_obj: Word dict with 'word', 'start', 'end', 'speaker'

        Returns:
            Smoothed speaker ID
        """
        raw_speaker = word_obj.get("speaker")
        word_start = word_obj.get("start", 0.0)
        word_end = word_obj.get("end", 0.0)

        # Add to history
        self.word_history.append({
            "word": word_obj.get("word", ""),
            "start": word_start,
            "end": word_end,
            "speaker": raw_speaker
        })

        # Trim history to window
        cutoff_time = word_end - (self.HISTORY_WINDOW_MS / 1000.0)
        self.word_history = [w for w in self.word_history if w["end"] >= cutoff_time]

        # Initialize current speaker if needed
        if self.current_speaker is None and raw_speaker is not None:
            self.current_speaker = raw_speaker
            self.current_speaker_start = word_start
            return self.current_speaker

        # If raw speaker matches current, continue
        if raw_speaker == self.current_speaker:
            return self.current_speaker

        # Potential speaker change detected
        if raw_speaker is not None and raw_speaker != self.current_speaker:
            # Calculate duration of current speaker
            duration_ms = (word_start - self.current_speaker_start) * 1000

            # Check if new speaker dominates recent history
            recent_speakers = [w["speaker"] for w in self.word_history if w["speaker"] is not None]
            if recent_speakers:
                new_speaker_ratio = recent_speakers.count(raw_speaker) / len(recent_speakers)

                # Accept speaker change if:
                # 1. Current speaker has been active long enough OR
                # 2. New speaker dominates recent history
                if (duration_ms >= self.MIN_SPEAKER_DURATION_MS or
                    new_speaker_ratio >= self.SPEAKER_CHANGE_THRESHOLD):
                    print(f"[SpeakerSmoother] Speaker change: {self.current_speaker} → {raw_speaker} "
                          f"(duration={duration_ms:.0f}ms, new_ratio={new_speaker_ratio:.2f})")
                    self.current_speaker = raw_speaker
                    self.current_speaker_start = word_start
                    return self.current_speaker

        # Default: keep current speaker (reject noise)
        return self.current_speaker


# Phase 3: Sentence boundary detection (legacy - not currently used)

class SentenceBoundaryDetector:
    """Hybrid sentence boundary detection using punctuation, pauses, and heuristics."""
    
    # Configuration
    PAUSE_SPLIT_MS = 1200        # Force split on long pause
    SOFT_PAUSE_HINT_MS = 400     # Hint if next word capitalized
    MAX_SENTENCE_SECS = 7.0      # Force split if too long
    MAX_SENTENCE_WORDS = 30      # Force split if too many words
    
    # Abbreviations that don't end sentences
    ABBREVS = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs", 
               "etc", "e.g", "i.e", "U.S", "a.m", "p.m", "Inc", "Ltd", "Corp"}
    
    # Terminal punctuation
    TERMINAL_PUNCT = {".", "?", "!"}
    
    @classmethod
    def should_split(cls, current_words: list[dict[str, Any]], next_word: dict[str, Any] | None) -> bool:
        """Determine if we should split a sentence at the current position."""
        if not current_words:
            return False
            
        last_word = current_words[-1]
        last_text = last_word.get("word", "").strip()
        last_end = last_word.get("end", 0.0)
        
        # Check 1: Max duration exceeded
        first_start = current_words[0].get("start", 0.0)
        duration = last_end - first_start
        if duration >= cls.MAX_SENTENCE_SECS:
            return True
        
        # Check 2: Max word count exceeded
        if len(current_words) >= cls.MAX_SENTENCE_WORDS:
            return True
        
        # Check 3: Terminal punctuation
        if any(last_text.endswith(p) for p in cls.TERMINAL_PUNCT):
            # Guard against abbreviations
            word_without_punct = last_text.rstrip(".?!")
            if word_without_punct in cls.ABBREVS:
                return False
            # Check for numbers/decimals (e.g., "3.14")
            if word_without_punct and word_without_punct[-1].isdigit():
                return False
            return True
        
        # Check 4: Long pause to next word
        if next_word:
            next_start = next_word.get("start", 0.0)
            pause_ms = (next_start - last_end) * 1000
            
            if pause_ms >= cls.PAUSE_SPLIT_MS:
                return True
            
            # Check 5: Soft pause + capitalization hint
            if pause_ms >= cls.SOFT_PAUSE_HINT_MS:
                next_text = next_word.get("word", "")
                if next_text and next_text[0].isupper():
                    return True
        
        return False


class SentenceAccumulator:
    """Accumulates words into sentences using boundary detection."""
    
    def __init__(self):
        self.current_words: list[dict[str, Any]] = []
        self.completed_sentences: list[dict[str, Any]] = []
    
    def add_word(self, word_obj: dict[str, Any]) -> dict[str, Any] | None:
        """Add a word and return a completed sentence if boundary detected."""
        self.current_words.append(word_obj)
        
        # Check if we should split (peek at next word, but we don't have it yet)
        # So we check without next_word for now
        if SentenceBoundaryDetector.should_split(self.current_words, None):
            return self.finalize_sentence()
        
        return None
    
    def finalize_sentence(self) -> dict[str, Any] | None:
        """Finalize current sentence and return it."""
        if not self.current_words:
            return None
        
        text = " ".join(w.get("word", "") for w in self.current_words)
        speaker = self.current_words[0].get("speaker")  # Use first word's speaker
        start = self.current_words[0].get("start", 0.0)
        end = self.current_words[-1].get("end", 0.0)
        
        sentence = {
            "text": text,
            "speaker": speaker,
            "start": start,
            "end": end,
            "words": self.current_words.copy()
        }
        
        print(f"[SentenceAccumulator] Finalized sentence: '{text}' ({len(self.current_words)} words)")
        
        self.completed_sentences.append(sentence)
        self.current_words = []
        
        return sentence
    
    def flush(self) -> dict[str, Any] | None:
        """Force finalize current sentence (e.g., on session end)."""
        return self.finalize_sentence()


class WindowManager:
    """Manages windowing of sentences for OpenAI processing."""
    
    # Configuration
    SENTENCE_TRIGGER = 2         # Send every N sentences (reduced from 3 for faster response)
    TIME_TRIGGER_SECS = 1.5      # Send every N seconds (reduced from 2.0 for faster response)
    OVERLAP_SIZE = 1             # Include last N sentences as overlap (reduced from 2)
    MAX_WINDOW_SECS = 30.0       # Safety cap
    
    def __init__(self):
        self.all_sentences: list[dict[str, Any]] = []
        self.last_send_time = 0.0
        self.window_id = 0
    
    def add_sentence(self, sentence: dict[str, Any]) -> dict[str, Any] | None:
        """Add a sentence and return a window if triggers fire."""
        import time
        
        self.all_sentences.append(sentence)
        current_time = time.time()
        
        # Initialize last_send_time on first sentence
        if self.last_send_time == 0.0:
            self.last_send_time = current_time
        
        # Check triggers
        sentence_count = len(self.all_sentences)
        time_since_last = current_time - self.last_send_time
        
        # Trigger 1: Sentence count
        if sentence_count >= self.SENTENCE_TRIGGER:
            return self.build_window(current_time)
        
        # Trigger 2: Time elapsed
        if time_since_last >= self.TIME_TRIGGER_SECS:
            return self.build_window(current_time)
        
        return None
    
    def build_window(self, current_time: float) -> dict[str, Any]:
        """Build a window with overlap."""
        self.window_id += 1
        
        # Determine overlap start index
        overlap_start = max(0, len(self.all_sentences) - self.SENTENCE_TRIGGER - self.OVERLAP_SIZE)
        
        # Get sentences for this window (overlap + new)
        window_sentences = self.all_sentences[overlap_start:]
        
        print(f"[WindowManager] Built window {self.window_id} with {len(window_sentences)} sentences")
        
        window = {
            "window_id": self.window_id,
            "sentences": window_sentences,
            "timestamp": current_time
        }
        
        # Keep only overlap sentences for next window
        self.all_sentences = self.all_sentences[-self.OVERLAP_SIZE:] if len(self.all_sentences) > self.OVERLAP_SIZE else []
        self.last_send_time = current_time
        
        return window


# Phase 4: Removed OpenAI integration (not needed - Deepgram handles filler words and formatting)

@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    session_id = str(uuid.uuid4())
    print(f"[session {session_id}] WebSocket connection attempt", flush=True)
    try:
        await websocket.accept()
        print(f"[session {session_id}] WebSocket accepted", flush=True)

        # Send a minimal ack so the frontend can verify send/receive
        await websocket.send_text(json.dumps({
            "type": "transcript",
            "text": f"session {session_id} connected",
            "is_final": False,
        }))
        print(f"[session {session_id}] Ack sent to client", flush=True)
    except Exception as e:
        print(f"[session {session_id}] ERROR in initial setup: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # Phase 2.5: Initialize speaker smoother and thought tracker
    speaker_smoother = SpeakerSmoother()
    thought_tracker = RealTimeThoughtTracker()

    print(f"[session {session_id}] Initialized components, connecting to Deepgram", flush=True)

    # Phase 2: Proxy to Deepgram real-time websocket
    api_key = os.getenv("DEEPGRAM_API_KEY")
    print(f"[session {session_id}] DEEPGRAM_API_KEY loaded: {bool(api_key)}")
    model = os.getenv("DG_MODEL", os.getenv("DEEPGRAM_MODEL", "nova-3"))  # Upgraded to nova-3
    language = os.getenv("DG_LANGUAGE", os.getenv("DEEPGRAM_LANGUAGE", "en"))

    # Using diarization mode for AI-based speaker detection
    diarize = os.getenv("DG_DIARIZE", os.getenv("DEEPGRAM_DIARIZE", "true")).lower() in {"1", "true", "yes"}
    interim = os.getenv("DG_INTERIM", os.getenv("DEEPGRAM_INTERIM", "true")).lower() in {"1", "true", "yes"}
    punctuate = os.getenv("DG_PUNCTUATE", os.getenv("DEEPGRAM_PUNCTUATE", "true")).lower() in {"1", "true", "yes"}

    print(f"[session {session_id}] Using DIARIZATION mode (mono audio, AI speaker detection)")
    num_channels = "1"
    use_diarize = "true" if diarize else "false"

    # Build Deepgram realtime URL with maximum accuracy parameters
    params = {
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": num_channels,
        "interim_results": "true" if interim else "false",
        "diarize": use_diarize,
        "diarize_version": "latest",  # Use latest diarization model for better accuracy
        "punctuate": "true" if punctuate else "false",
        "smart_format": "true",
        "language": language,
        "model": model,
        "filler_words": "true",      # Better handling of um, uh, etc
        "numerals": "true",           # Convert numbers to digits
        "profanity_filter": "false",  # Don't censor (need accuracy)
        "redact": "false",            # No PII redaction
        "utterances": "true",         # Better utterance boundaries
        "endpointing": "300",         # 300ms silence to detect end of speech
        "vad_events": "true",         # Voice activity detection events
        "paragraphs": "true",         # Better paragraph formatting
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    dg_url = f"wss://api.deepgram.com/v1/listen?{query}"

    if not api_key:
        # No key available; signal error and close
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Deepgram API key not configured",
        }))
        await websocket.close()
        print(f"[session {session_id}] missing DEEPGRAM_API_KEY; closed")
        return

    import websockets

    deepgram_ws: websockets.WebSocketClientProtocol | None = None
    client_to_dg_task: asyncio.Task | None = None
    dg_to_client_task: asyncio.Task | None = None
    bytes_received = 0

    async def forward_client_audio_to_deepgram() -> None:
        nonlocal bytes_received
        while True:
            try:
                data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                print(f"[session {session_id}] Client disconnected in audio forward")
                break
            except Exception as e:
                print(f"[session {session_id}] Error receiving audio from client: {e}")
                break
            bytes_received += len(data)
            try:
                if deepgram_ws is not None:
                    await deepgram_ws.send(data)
                else:
                    print(f"[session {session_id}] Deepgram WS is None, stopping")
                    break
            except Exception as e:
                print(f"[session {session_id}] Error sending to Deepgram: {e}")
                break
        print(f"[session {session_id}] Audio forward loop ended, sent {bytes_received} bytes total")

    async def map_deepgram_message(raw: str) -> dict[str, Any] | None:
        nonlocal speaker_smoother, thought_tracker  # Access the outer scope variables
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        
        # Don't log every Results message (too noisy), just track them
        # if payload.get("type") == "Results":
        #     print(f"[session {session_id}] Deepgram Results: {json.dumps(payload, indent=2)[:500]}...")
        # Expect Deepgram messages with type "Results"
        if payload.get("type") != "Results":
            # Forward non-results as debug to help diagnose connection state
            try:
                return {"type": "transcript", "text": f"[dg:{payload.get('type')}]", "is_final": False}
            except Exception:
                return None
        is_final = bool(payload.get("is_final"))
        channel = payload.get("channel") or {}
        alts = channel.get("alternatives") or []
        text = ""
        words_data = []
        speaker = None
        
        if alts:
            text = alts[0].get("transcript") or ""
            # Extract word-level data with speaker labels and timestamps
            # Use RAW speaker labels from Deepgram (no smoothing for better accuracy)
            words = alts[0].get("words") or []
            for word_obj in words:
                # Use raw speaker from Deepgram
                raw_speaker = word_obj.get("speaker")

                word_entry = {
                    "word": word_obj.get("word", ""),
                    "start": word_obj.get("start", 0.0),
                    "end": word_obj.get("end", 0.0),
                    "speaker": raw_speaker  # Use raw speaker from Deepgram
                }
                words_data.append(word_entry)

            # Determine dominant speaker for this segment (using raw data)
            if words_data:
                speakers = [w["speaker"] for w in words_data if w["speaker"] is not None]
                if speakers:
                    # Use most common speaker in this segment
                    speaker = max(set(speakers), key=speakers.count)
        
        # Phase 3: Track thoughts in real-time
        if is_final and text.strip() and speaker is not None:
            print(f"[session {session_id}] Final transcript - text: '{text}', speaker: {speaker}", flush=True)

            # Get timestamp from words if available
            start_ms = 0
            end_ms = 0
            if words_data and len(words_data) > 0:
                start_ms = int(words_data[0].get("start", 0.0) * 1000)
                end_ms = int(words_data[-1].get("end", 0.0) * 1000)

            # Add to thought tracker - this merges same-speaker segments automatically
            finalized_thought, current_thought = thought_tracker.add_segment(speaker, text, start_ms, end_ms)

            # If previous speaker's thought was finalized, send it
            if finalized_thought:
                await websocket.send_text(json.dumps({
                    "type": "thought_update",
                    "thought_id": finalized_thought["id"],
                    "sequence": finalized_thought["sequence"],
                    "speaker": finalized_thought["speaker"],
                    "text": finalized_thought["text"],
                    "start_ms": finalized_thought["start_ms"],
                    "end_ms": finalized_thought["end_ms"],
                    "is_final": finalized_thought["is_final"]
                }))

            # Send current thought immediately
            await websocket.send_text(json.dumps({
                "type": "thought_update",
                "thought_id": current_thought["id"],
                "sequence": current_thought["sequence"],
                "speaker": current_thought["speaker"],
                "text": current_thought["text"],
                "start_ms": current_thought["start_ms"],
                "end_ms": current_thought["end_ms"],
                "is_final": current_thought["is_final"]
            }))
        
        result = {
            "type": "transcript",
            "text": text,
            "is_final": is_final
        }
        
        # Only include words and speaker if we have diarization data
        if words_data:
            result["words"] = words_data
        if speaker is not None:
            result["speaker"] = speaker
        
        return result

    async def forward_deepgram_results_to_client() -> None:
        if deepgram_ws is None:
            print(f"[session {session_id}] Cannot forward DG results, deepgram_ws is None")
            return
        message_count = 0
        try:
            async for message in deepgram_ws:
                message_count += 1
                if isinstance(message, (bytes, bytearray)):
                    # ignore binary messages from DG (unlikely here)
                    continue
                mapped = await map_deepgram_message(message)
                if mapped is not None:
                    await websocket.send_text(json.dumps(mapped))
        except Exception as e:
            # Will be handled by outer finally/cleanup
            print(f"[session {session_id}] Error in forward_deepgram_results_to_client after {message_count} messages: {e}")
            pass
        finally:
            print(f"[session {session_id}] Deepgram results loop ended, processed {message_count} messages")

    try:
        print(f"[session {session_id}] Attempting to connect to Deepgram...", flush=True)
        deepgram_ws = await websockets.connect(
            dg_url,
            extra_headers={"Authorization": f"Token {api_key}"},
            max_size=2 ** 23,
            ping_interval=20,
            ping_timeout=20,
        )
        print(f"[session {session_id}] connected to Deepgram url={dg_url}", flush=True)
        # Notify client that backend is connected to Deepgram and listening
        try:
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": "listening...",
                "is_final": False,
            }))
        except Exception:
            pass
        client_to_dg_task = asyncio.create_task(forward_client_audio_to_deepgram())
        dg_to_client_task = asyncio.create_task(forward_deepgram_results_to_client())

        # Wait for either side to finish
        done, pending = await asyncio.wait(
            {client_to_dg_task, dg_to_client_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        print(f"[session {session_id}] client disconnected", flush=True)
    except Exception as exc:
        print(f"[session {session_id}] Deepgram/connect error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Deepgram error: {exc}",
            }))
        except Exception:
            pass
    finally:
        if deepgram_ws is not None:
            try:
                await deepgram_ws.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
        print(f"[session {session_id}] closed - bytes={bytes_received}")


def get_port() -> int:
    try:
        return int(os.getenv("PORT", "8080"))
    except ValueError:
        return 8080


if __name__ == "__main__":
    import uvicorn
    # Ensure project root is importable in the reload subprocess
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    uvicorn.run("server.main:app", host="0.0.0.0", port=get_port(), reload=True, reload_dirs=[str(project_root)])


