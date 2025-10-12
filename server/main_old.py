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


# Real-time speaker-aware thought tracking

class SpeakerThoughtTracker:
    """Hybrid sentence boundary detection using punctuation, pauses, and heuristics."""

    # Configuration
    PAUSE_SPLIT_MS = 800         # Force split on long pause (reduced from 1200ms)
    SOFT_PAUSE_HINT_MS = 400     # Hint if next word capitalized
    MAX_SENTENCE_SECS = 3.0      # Force split if too long (reduced from 7.0s)
    MAX_SENTENCE_WORDS = 15      # Force split if too many words (reduced from 30)
    
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
        should_split = SentenceBoundaryDetector.should_split(self.current_words, None)
        if should_split:
            print(f"[SentenceAccumulator] Boundary detected after adding word '{word_obj.get('word', '')}' (total words: {len(self.current_words)})")
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
    SENTENCE_TRIGGER = 3         # Send every N sentences (increased from 2 to allow more accumulation)
    TIME_TRIGGER_SECS = 3.0      # Send every N seconds (increased from 1.5 to allow more accumulation)
    OVERLAP_SIZE = 1             # Include last N sentences as overlap
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

        print(f"[WindowManager] Added sentence. Count: {sentence_count}/{self.SENTENCE_TRIGGER}, Time since last: {time_since_last:.1f}s/{self.TIME_TRIGGER_SECS}s")

        # Trigger 1: Sentence count
        if sentence_count >= self.SENTENCE_TRIGGER:
            print(f"[WindowManager] Sentence count trigger fired ({sentence_count} >= {self.SENTENCE_TRIGGER})")
            return self.build_window(current_time)

        # Trigger 2: Time elapsed
        if time_since_last >= self.TIME_TRIGGER_SECS:
            print(f"[WindowManager] Time trigger fired ({time_since_last:.1f}s >= {self.TIME_TRIGGER_SECS}s)")
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


# Phase 4: OpenAI integration for structured output

async def process_sentences_into_thoughts(
    sentences: list[dict[str, Any]],
    session_id: str,
    prior_segments: list[dict[str, Any]] = None,
    max_segments: int = 5,
    min_gap_ms: int = 1200,
    include_keywords: bool = True
) -> dict[str, Any] | None:
    """
    Process finalized sentences with OpenAI to group them into coherent 'thoughts'.

    Args:
        sentences: List of sentence dicts with {text, start, end, speaker}
        session_id: Current session ID for logging
        prior_segments: Optional list of recent segment summaries for context
        max_segments: Maximum number of thought segments to create
        min_gap_ms: Minimum gap in ms to prefer a new thought boundary
        include_keywords: Whether to include keywords in output

    Returns:
        Structured output dictionary with thought segments or None if OpenAI fails
    """
    import openai

    print(f"[session {session_id}] ===== OPENAI PROCESSING STARTED =====")
    print(f"[session {session_id}] Processing {len(sentences)} sentences with OpenAI")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key or openai_key == "your_openai_key_here":
        print(f"[session {session_id}] ERROR: No OpenAI API key configured, skipping structured output")
        return None

    print(f"[session {session_id}] OpenAI API key found: {openai_key[:20]}...")

    if not sentences:
        print(f"[session {session_id}] No sentences to process, returning empty segments")
        return {"segments": []}
    
    # Convert sentences to the format expected by OpenAI (with milliseconds and speaker)
    formatted_sentences = []
    for sent in sentences:
        speaker = sent.get("speaker")
        speaker_label = f"Speaker {speaker}" if speaker is not None else "Unknown"
        formatted_sentences.append({
            "speaker": speaker_label,
            "text": sent.get("text", ""),
            "start_ms": int(sent.get("start", 0.0) * 1000),
            "end_ms": int(sent.get("end", 0.0) * 1000)
        })
    
    # Format prior segments for context (if any)
    prior_context = prior_segments if prior_segments else []
    
    # Build the system prompt
    system_prompt = """You are a transcript post-processor for INTERVIEW recordings.
Your job is to group transcript sentences into complete interview exchanges (question + answer pairs).

CONTEXT: This is an INTERVIEW conversation between an interviewer and candidate.

Rules:
1. INTERVIEW GROUPING (MOST IMPORTANT): Group question-answer pairs together as ONE thought
   - When one speaker asks a question, include the ENTIRE response from the other speaker in the same thought
   - Example: "Speaker 0: Tell me about yourself. Speaker 1: I have 5 years of experience in software..." = ONE thought
   - Do NOT split based on speaker changes - interviews are back-and-forth dialogue
2. Do not invent timestamps. Use the provided start_ms and end_ms from input sentences only (span from first sentence to last in the group).
3. A thought should represent a complete interview exchange or a complete topic discussion between speakers.
4. Clean up filler words (um, uh, like) and fix obvious grammar issues while preserving meaning.
5. Format multi-speaker thoughts clearly with each speaker labeled: "Speaker 0: [text]\nSpeaker 1: [text]"
6. Only start a NEW thought when the conversation moves to a completely different topic or question.
7. Output valid JSON that conforms exactly to the provided schema. No extra text.
8. If input is empty, return {"segments": []}.
"""
    
    # Build the user prompt
    user_prompt = f"""Group the following finalized transcript sentences into thought segments. Use the JSON schema below.
Use the following constraints to guide boundaries:

max_segments: {max_segments}
min_gap_ms_for_new_thought: {min_gap_ms} (if a gap between consecutive sentences ≥ this value, prefer a new thought)

Prior context (optional, may be empty):
prior_segments: {json.dumps(prior_context)}
(Each object has {{id, text}}; use only to maintain continuity—do not copy timestamps.)

Sentences (final only):
{json.dumps(formatted_sentences, indent=2)}

JSON Schema (must conform exactly):
{{
  "type": "object",
  "required": ["segments"],
  "properties": {{
    "segments": {{
      "type": "array",
      "items": {{
        "type": "object",
        "required": ["id","speaker","text","start_ms","end_ms"],
        "properties": {{
          "id": {{ "type": "string" }},
          "speaker": {{ "type": "string" }},
          "text": {{ "type": "string" }},
          "start_ms": {{ "type": "integer", "minimum": 0 }},
          "end_ms": {{ "type": "integer", "minimum": 0 }}
        }}
      }}
    }}
  }}
}}

Output:
Return only a JSON object that matches the schema. No narration."""

    try:
        print(f"[session {session_id}] Creating OpenAI client...")
        client = openai.AsyncOpenAI(api_key=openai_key)

        print(f"[session {session_id}] Sending request to OpenAI (model: gpt-4o-mini)...")
        print(f"[session {session_id}] Request payload preview - sentences: {len(formatted_sentences)}, prior_segments: {len(prior_context)}")

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent structure
        )

        print(f"[session {session_id}] OpenAI response received!")

        result_text = response.choices[0].message.content
        if result_text:
            print(f"[session {session_id}] Parsing OpenAI response (length: {len(result_text)} chars)...")
            structured_output = json.loads(result_text)
            segments_count = len(structured_output.get('segments', []))
            print(f"[session {session_id}] ✓ SUCCESS: OpenAI grouped {len(sentences)} sentences into {segments_count} thoughts")
            print(f"[session {session_id}] Segments: {json.dumps(structured_output.get('segments', []), indent=2)}")
            print(f"[session {session_id}] ===== OPENAI PROCESSING COMPLETE =====")
            return structured_output
        else:
            print(f"[session {session_id}] WARNING: OpenAI returned empty response")

    except openai.APIError as e:
        print(f"[session {session_id}] ✗ OpenAI API error: {e}")
        print(f"[session {session_id}] Error details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[session {session_id}] ✗ Unexpected error during OpenAI processing: {e}")
        print(f"[session {session_id}] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

    return None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/debug/openai")
async def debug_openai() -> dict[str, Any]:
    """Debug endpoint to test OpenAI integration with mock data."""
    import uuid

    session_id = f"debug-{uuid.uuid4()}"

    # Mock sentences for testing
    test_sentences = [
        {
            "text": "Hello, how are you doing today?",
            "speaker": 0,
            "start": 0.0,
            "end": 2.5,
            "words": []
        },
        {
            "text": "I'm doing great, thanks for asking.",
            "speaker": 1,
            "start": 3.0,
            "end": 5.0,
            "words": []
        },
        {
            "text": "That's wonderful to hear.",
            "speaker": 0,
            "start": 5.5,
            "end": 7.0,
            "words": []
        }
    ]

    print(f"[{session_id}] Testing OpenAI with mock data...")

    result = await process_sentences_into_thoughts(
        sentences=test_sentences,
        session_id=session_id,
        prior_segments=None,
        max_segments=5,
        min_gap_ms=1200,
        include_keywords=True
    )

    if result:
        return {
            "status": "success",
            "message": "OpenAI integration working",
            "result": result
        }
    else:
        return {
            "status": "error",
            "message": "OpenAI integration failed - check server logs"
        }


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    session_id = str(uuid.uuid4())
    print(f"[session {session_id}] WebSocket endpoint hit, attempting to accept...")
    try:
        await websocket.accept()
        print(f"[session {session_id}] WebSocket accepted successfully")

        # Send a minimal ack so the frontend can verify send/receive
        await websocket.send_text(json.dumps({
            "type": "transcript",
            "text": f"session {session_id} connected",
            "is_final": False,
        }))
        print(f"[session {session_id}] Ack sent to client")
    except Exception as e:
        print(f"[session {session_id}] ERROR in initial setup: {e}")
        import traceback
        traceback.print_exc()
        return

    # Phase 3: Initialize sentence accumulation and windowing
    sentence_accumulator = SentenceAccumulator()
    window_manager = WindowManager()
    
    # Phase 4: Track prior segments for OpenAI context
    prior_segments: list[dict[str, Any]] = []  # Keep last N segment summaries
    
    # Phase 2: Proxy to Deepgram real-time websocket
    api_key = os.getenv("DEEPGRAM_API_KEY")
    print(f"[session {session_id}] DEEPGRAM_API_KEY loaded: {bool(api_key)}")
    model = os.getenv("DG_MODEL", os.getenv("DEEPGRAM_MODEL", "nova-2"))
    language = os.getenv("DG_LANGUAGE", os.getenv("DEEPGRAM_LANGUAGE", "en"))
    diarize = os.getenv("DG_DIARIZE", os.getenv("DEEPGRAM_DIARIZE", "true")).lower() in {"1", "true", "yes"}
    interim = os.getenv("DG_INTERIM", os.getenv("DEEPGRAM_INTERIM", "true")).lower() in {"1", "true", "yes"}
    punctuate = os.getenv("DG_PUNCTUATE", os.getenv("DEEPGRAM_PUNCTUATE", "true")).lower() in {"1", "true", "yes"}




    # Build Deepgram realtime URL
    params = {
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "interim_results": "true" if interim else "false",
        "diarize": "true" if diarize else "false",
        "punctuate": "true" if punctuate else "false",
        "smart_format": "true",
        "language": language,
        "model": model,
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
        nonlocal prior_segments  # Access the outer scope variable
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
            words = alts[0].get("words") or []
            for word_obj in words:
                word_entry = {
                    "word": word_obj.get("word", ""),
                    "start": word_obj.get("start", 0.0),
                    "end": word_obj.get("end", 0.0),
                    "speaker": word_obj.get("speaker")
                }
                words_data.append(word_entry)
            
            # Determine dominant speaker for this segment
            if words_data:
                speakers = [w["speaker"] for w in words_data if w["speaker"] is not None]
                if speakers:
                    # Use most common speaker in this segment
                    speaker = max(set(speakers), key=speakers.count)
        
        # Phase 3: Feed words into sentence accumulator for final transcripts
        if is_final:
            print(f"[session {session_id}] ===== FINAL TRANSCRIPT RECEIVED =====")
            print(f"[session {session_id}] Final transcript - text: '{text}', words_data: {len(words_data)} words, speaker: {speaker}")
            if not words_data:
                print(f"[session {session_id}] WARNING: No words_data in final transcript!")

            if words_data:
                print(f"[session {session_id}] Processing {len(words_data)} words from final transcript")
                for idx, word_obj in enumerate(words_data):
                    # Reduced logging - only log every 5th word to avoid spam
                    if idx % 5 == 0:
                        print(f"[session {session_id}] Word {idx+1}/{len(words_data)}: '{word_obj.get('word', '')}' (speaker: {word_obj.get('speaker')})")
                    completed_sentence = sentence_accumulator.add_word(word_obj)
                    if completed_sentence:
                        # Sentence boundary detected, pass to window manager
                        print(f"[session {session_id}] ✓ Sentence completed: '{completed_sentence.get('text', '')}'")
                        window = window_manager.add_sentence(completed_sentence)
                        if window:
                            # Window trigger fired - process with OpenAI (Phase 4)
                            sentences = window['sentences']
                            print(f"[session {session_id}] ===== WINDOW TRIGGER FIRED =====")
                            print(f"[session {session_id}] Window ID: {window['window_id']} with {len(sentences)} sentences")

                            # Process sentences into thoughts with OpenAI
                            structured_output = await process_sentences_into_thoughts(
                                sentences=sentences,
                                session_id=session_id,
                                prior_segments=prior_segments[-3:] if prior_segments else None,  # Last 3 segments for context
                                max_segments=5,
                                min_gap_ms=1200,
                                include_keywords=True
                            )

                            if structured_output and structured_output.get("segments"):
                                # Update prior_segments with new segment text
                                for seg in structured_output["segments"]:
                                    prior_segments.append({
                                        "id": seg.get("id"),
                                        "text": seg.get("text")
                                    })
                                # Keep only last 5 segments for context
                                if len(prior_segments) > 5:
                                    prior_segments = prior_segments[-5:]

                                print(f"[session {session_id}] Sending {len(structured_output.get('segments', []))} thoughts to client")
                                # Send structured output to client
                                await websocket.send_text(json.dumps({
                                    "type": "thoughts",
                                    "data": structured_output
                                }))
                                print(f"[session {session_id}] ✓ Thoughts sent to client successfully")
                            else:
                                print(f"[session {session_id}] ✗ No structured output from OpenAI")
        
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
        deepgram_ws = await websockets.connect(
            dg_url,
            extra_headers={"Authorization": f"Token {api_key}"},
            max_size=2 ** 23,
            ping_interval=20,
            ping_timeout=20,
        )
        print(f"[session {session_id}] connected to Deepgram url={dg_url}")
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
        print(f"[session {session_id}] client disconnected")
    except Exception as exc:
        print(f"[session {session_id}] Deepgram/connect error: {exc}")
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


