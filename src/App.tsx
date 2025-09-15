import React from 'react';
import { createMicrophonePcmStream } from './lib/audio';
import { createWebSocketClient, ServerMessage } from './lib/ws';

function useTranscript() {
  const [partial, setPartial] = React.useState('');
  const [finals, setFinals] = React.useState<string[]>([]);

  function handleMessage(msg: ServerMessage) {
    if (msg.type === 'transcript') {
      if (msg.is_final) {
        setFinals((prev) => [...prev, msg.text]);
        setPartial('');
      } else {
        setPartial(msg.text);
      }
    }
  }

  return { partial, finals, handleMessage };
}

export default function App() {
  const [wsUrl, setWsUrl] = React.useState<string>(
    (import.meta as any).env?.VITE_WS_URL || 'ws://localhost:8080/stream'
  );
  const [status, setStatus] = React.useState<'idle' | 'connecting' | 'recording' | 'stopped' | 'error'>('idle');
  const [log, setLog] = React.useState<string>('');
  const { partial, finals, handleMessage } = useTranscript();

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
        <textarea readOnly value={[...finals, partial].filter(Boolean).join('\n')} />
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Log</h3>
        <div className="log">{log || 'No logs yet.'}</div>
      </div>
    </div>
  );
}
