import React from 'react';
import { createMicrophonePcmStream } from './lib/audio';
import { createWebSocketClient, ServerMessage, ThoughtSegment } from './lib/ws';

type LiveThought = {
  speaker: number;
  text: string;
  startTime: number;
  lastUpdateTime: number;
  isInterim: boolean;  // Track if this is still being updated
};

type CleanedThought = {
  id: string;
  speaker: string;
  text: string;
  start_ms: number;
  end_ms: number;
  sequence: number;
  is_final: boolean;
};

function useTranscript() {
  const [partial, setPartial] = React.useState('');
  const [finals, setFinals] = React.useState<string[]>([]);
  const [liveThoughts, setLiveThoughts] = React.useState<LiveThought[]>([]);
  const [cleanedThoughts, setCleanedThoughts] = React.useState<CleanedThought[]>([]);

  function handleMessage(msg: ServerMessage) {
    if (msg.type === 'transcript') {
      const speaker = msg.speaker !== undefined ? msg.speaker : -1;
      const text = msg.text;
      
      // Extract timestamp from words data if available
      let timestamp = 0;
      if (msg.words && msg.words.length > 0) {
        timestamp = msg.words[0].start;
      }
      
      // For raw transcript debug view
      const speakerLabel = speaker !== -1 ? `Speaker ${speaker}: ` : '';
      const timestampStr = timestamp > 0 ? `[${timestamp.toFixed(1)}s] ` : '';
      const formattedText = timestampStr + speakerLabel + text;
      
      // Skip empty messages or unknown speakers
      if (!text.trim() || speaker === -1) {
        return;
      }
      
      if (msg.is_final) {
        setFinals((prev) => [...prev, formattedText]);
        setPartial('');
        
        // Update live thoughts - accumulate by speaker (FINAL)
        setLiveThoughts((prev) => {
          const lastThought = prev[prev.length - 1];
          
          // If same speaker and was interim, finalize it (REPLACE the interim text)
          if (lastThought && lastThought.speaker === speaker && lastThought.isInterim) {
            return [
              ...prev.slice(0, -1),
              {
                ...lastThought,
                text: text,  // REPLACE interim with final, don't append
                lastUpdateTime: timestamp || lastThought.lastUpdateTime,
                isInterim: false  // Mark as finalized
              }
            ];
          } else if (lastThought && lastThought.speaker === speaker && !lastThought.isInterim) {
            // Same speaker, already final - append
            return [
              ...prev.slice(0, -1),
              {
                ...lastThought,
                text: lastThought.text + ' ' + text,
                lastUpdateTime: timestamp || lastThought.lastUpdateTime,
                isInterim: false
              }
            ];
          } else {
            // New speaker or first thought - create new thought
            const startTime = timestamp > 0 ? timestamp : (lastThought ? lastThought.lastUpdateTime + 1 : 0);
            return [
              ...prev,
              {
                speaker,
                text,
                startTime,
                lastUpdateTime: startTime,
                isInterim: false
              }
            ];
          }
        });
      } else {
        // INTERIM results - update live thoughts in real-time
        setPartial(formattedText);
        
        setLiveThoughts((prev) => {
          const lastThought = prev[prev.length - 1];
          
          // If same speaker and interim, UPDATE (replace) the text
          if (lastThought && lastThought.speaker === speaker && lastThought.isInterim) {
            return [
              ...prev.slice(0, -1),
              {
                ...lastThought,
                text: text,  // REPLACE, don't append for interim
                lastUpdateTime: timestamp || lastThought.lastUpdateTime,
                isInterim: true
              }
            ];
          } else if (lastThought && lastThought.speaker === speaker && !lastThought.isInterim) {
            // Same speaker but previous was final - create new interim thought
            const startTime = timestamp > 0 ? timestamp : lastThought.lastUpdateTime;
            return [
              ...prev,
              {
                speaker,
                text,
                startTime,
                lastUpdateTime: startTime,
                isInterim: true
              }
            ];
          } else {
            // New speaker - create new interim thought
            const startTime = timestamp > 0 ? timestamp : (lastThought ? lastThought.lastUpdateTime + 1 : 0);
            return [
              ...prev,
              {
                speaker,
                text,
                startTime,
                lastUpdateTime: startTime,
                isInterim: true
              }
            ];
          }
        });
      }
    } else if (msg.type === 'thoughts') {
      // OpenAI cleaned thoughts (old format, for comparison)
      console.log('[App] Received thoughts:', msg.data.segments);
      setCleanedThoughts((prev) => [...prev, ...msg.data.segments.map(s => ({
        ...s,
        sequence: 0,
        is_final: true
      }))]);
    } else if (msg.type === 'thought_update') {
      // Real-time thought updates with sequence tracking
      console.log('[App] Received thought_update:', msg.thought_id, 'seq:', msg.sequence);
      setCleanedThoughts((prev) => {
        const existingIndex = prev.findIndex(t => t.id === msg.thought_id);

        if (existingIndex >= 0) {
          const existing = prev[existingIndex];
          // Only update if sequence is newer or equal
          if (msg.sequence >= existing.sequence) {
            const updated = [...prev];
            updated[existingIndex] = {
              id: msg.thought_id,
              speaker: msg.speaker,
              text: msg.text,
              start_ms: msg.start_ms,
              end_ms: msg.end_ms,
              sequence: msg.sequence,
              is_final: msg.is_final
            };
            return updated;
          }
          return prev;
        } else {
          // New thought
          return [...prev, {
            id: msg.thought_id,
            speaker: msg.speaker,
            text: msg.text,
            start_ms: msg.start_ms,
            end_ms: msg.end_ms,
            sequence: msg.sequence,
            is_final: msg.is_final
          }];
        }
      });
    }
  }

  return { partial, finals, liveThoughts, cleanedThoughts, handleMessage };
}

export default function App() {
  const [wsUrl, setWsUrl] = React.useState<string>(
    (import.meta as any).env?.VITE_WS_URL || 'ws://localhost:8080/stream'
  );
  const [status, setStatus] = React.useState<'idle' | 'connecting' | 'recording' | 'stopped' | 'error'>('idle');
  const [log, setLog] = React.useState<string>('');
  const { partial, finals, liveThoughts, cleanedThoughts, handleMessage } = useTranscript();

  const wsRef = React.useRef<ReturnType<typeof createWebSocketClient> | null>(null);
  const micRef = React.useRef<ReturnType<typeof createMicrophonePcmStream> | null>(null);

  function appendLog(line: string) {
    setLog((l) => `${l}${l ? '\n' : ''}${new Date().toLocaleTimeString()} - ${line}`);
  }

  async function connect() {
    if (wsRef.current?.isOpen()) return;
    setStatus('connecting');
    appendLog(`Connecting to ${wsUrl}`);
    wsRef.current = createWebSocketClient(wsUrl, (msg) => {
      handleMessage(msg);
    }, (err) => {
      console.error(err);
      setStatus('error');
      appendLog(`WebSocket error: ${String(err)}`);
    });
    try {
      await wsRef.current.connect();
      appendLog('Connected');
    } catch (err) {
      setStatus('error');
      appendLog(`Failed to connect: ${String(err)}`);
      return;
    }
  }

  async function startRecording() {
    await connect();
    if (!wsRef.current?.isOpen()) return;
    if (micRef.current?.isRecording()) return;

    micRef.current = createMicrophonePcmStream({
      onChunk: (chunk) => {
        wsRef.current?.sendAudioChunk(chunk);
      },
      onError: (err) => {
        setStatus('error');
        appendLog(`Audio error: ${String(err)}`);
      }
    });

    try {
      await micRef.current.start();
      setStatus('recording');
      appendLog('Recording started');
    } catch (err) {
      setStatus('error');
      appendLog(`Mic start failed: ${String(err)}`);
    }
  }

  function stopAll() {
    micRef.current?.stop();
    wsRef.current?.close();
    setStatus('stopped');
    appendLog('Stopped');
  }

  return (
    <div className="container">
      <header>
        <h1>Interview Streamer</h1>
        <div className="status">Status: {status}</div>
      </header>

      <div className="card">
        <div className="small">WebSocket URL</div>
        <input
          style={{ width: '100%', padding: 8, borderRadius: 8, border: '1px solid color-mix(in oklab, canvas, canvasText 12%)', background: 'transparent', color: 'inherit' }}
          value={wsUrl}
          onChange={(e) => setWsUrl(e.target.value)}
          placeholder="ws://host:port/stream"
        />
        <div className="controls">
          <button className="primary" onClick={startRecording} disabled={status === 'recording'}>Start</button>
          <button onClick={stopAll} disabled={status !== 'recording' && status !== 'connecting'}>Stop</button>
        </div>
        <div className="badge">PCM16 • 16kHz • 250ms frames</div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Live Transcript</h3>
        {liveThoughts.length === 0 && !partial ? (
          <div style={{ color: '#999', fontStyle: 'italic', padding: '20px 0' }}>
            Waiting for speech...
          </div>
        ) : (
          <textarea
            readOnly
            value={liveThoughts.map((thought) => {
              const formatTime = (seconds: number) => {
                const totalSeconds = Math.floor(seconds);
                const hours = Math.floor(totalSeconds / 3600);
                const minutes = Math.floor((totalSeconds % 3600) / 60);
                const secs = totalSeconds % 60;
                return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
              };
              return `[${formatTime(thought.startTime)}] Speaker ${thought.speaker}: ${thought.text}`;
            }).join('\n')}
            style={{ width: '100%', minHeight: '200px', fontFamily: 'monospace' }}
          />
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Organized Transcript ({cleanedThoughts.length} thoughts)</h3>
        {cleanedThoughts.length === 0 ? (
          <div style={{ color: '#999', fontStyle: 'italic', padding: '20px 0' }}>
            Waiting for speech segments...
          </div>
        ) : (
          cleanedThoughts.map((thought) => {
            const formatTime = (ms: number) => {
              const totalSeconds = Math.floor(ms / 1000);
              const hours = Math.floor(totalSeconds / 3600);
              const minutes = Math.floor((totalSeconds % 3600) / 60);
              const seconds = totalSeconds % 60;
              return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            };

            // Split text by \n\n to create paragraphs
            const paragraphs = thought.text.split('\n\n').filter(p => p.trim());

            return (
              <div
                key={thought.id}
                style={{
                  marginBottom: 16,
                  padding: 12,
                  background: 'rgba(0,0,0,0.03)',
                  borderRadius: 6,
                  transition: 'all 0.3s ease-in-out',
                  opacity: thought.is_final ? 1 : 0.85
                }}
              >
                <div style={{ marginBottom: 8 }}>
                  <span style={{ color: '#666', fontSize: '0.9em' }}>
                    [{formatTime(thought.start_ms)} - {formatTime(thought.end_ms)}]
                  </span>
                  {' '}
                  <strong>{thought.speaker}:</strong>
                  {!thought.is_final && <span style={{ color: '#999', fontSize: '0.85em', marginLeft: 8 }}>(updating...)</span>}
                </div>
                {paragraphs.map((para, idx) => (
                  <p key={idx} style={{
                    margin: '8px 0',
                    lineHeight: 1.6,
                    transition: 'opacity 0.3s ease-in-out'
                  }}>
                    {para}
                  </p>
                ))}
              </div>
            );
          })
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Log</h3>
        <div className="log">{log || 'No logs yet.'}</div>
      </div>
    </div>
  );
}
