export type TranscriptMessage = {
  type: 'transcript';
  text: string;
  is_final?: boolean;
};

export type ServerMessage = TranscriptMessage | { type: 'error'; message: string };

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
