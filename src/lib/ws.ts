export type WordData = {
  word: string;
  start: number;
  end: number;
  speaker: number | null;
};

export type TranscriptMessage = {
  type: 'transcript';
  text: string;
  is_final?: boolean;
  words?: WordData[];
  speaker?: number;
};

export type ThoughtSegment = {
  id: string;
  speaker: string;
  text: string;
  start_ms: number;
  end_ms: number;
};

export type ThoughtsMessage = {
  type: 'thoughts';
  data: {
    segments: ThoughtSegment[];
  };
};

export type ThoughtUpdateMessage = {
  type: 'thought_update';
  thought_id: string;
  sequence: number;
  speaker: string;
  text: string;
  start_ms: number;
  end_ms: number;
  is_final: boolean;
};

export type ServerMessage = TranscriptMessage | ThoughtsMessage | ThoughtUpdateMessage | { type: 'error'; message: string };

export type WebSocketClient = {
  connect: () => Promise<void>;
  sendAudioChunk: (chunk: ArrayBufferView | ArrayBufferLike) => void;
  close: () => void;
  isOpen: () => boolean;
};

export function createWebSocketClient(url: string, onMessage: (msg: ServerMessage) => void, onError?: (err: unknown) => void): WebSocketClient {
  let socket: WebSocket | null = null;
  let openPromise: Promise<void> | null = null;

  function connect(): Promise<void> {
    if (socket && socket.readyState === WebSocket.OPEN) return Promise.resolve();
    if (openPromise) return openPromise;
    openPromise = new Promise<void>((resolve, reject) => {
      try {
        socket = new WebSocket(url);
        socket.binaryType = 'arraybuffer';
        socket.onopen = () => resolve();
        socket.onerror = (ev) => {
          onError?.(ev);
          reject(new Error('WebSocket error'));
        };
        socket.onclose = () => {
          socket = null;
          openPromise = null;
        };
        socket.onmessage = (ev) => {
          try {
            if (typeof ev.data === 'string') {
              const msg = JSON.parse(ev.data) as ServerMessage;
              onMessage(msg);
            }
          } catch (err) {
            onError?.(err);
          }
        };
      } catch (err) {
        onError?.(err);
        reject(err);
      }
    });
    return openPromise;
  }

  function sendAudioChunk(chunk: ArrayBufferView | ArrayBufferLike) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      if (ArrayBuffer.isView(chunk)) {
        socket.send(chunk);
      } else {
        const view = new Uint8Array(chunk);
        socket.send(view);
      }
    }
  }

  function close() {
    if (socket) {
      socket.close();
      socket = null;
      openPromise = null;
    }
  }

  function isOpen() {
    return !!socket && socket.readyState === WebSocket.OPEN;
  }

  return { connect, sendAudioChunk, close, isOpen };
}
