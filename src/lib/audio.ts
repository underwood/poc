export type AudioStreamCallbacks = {
  onChunk: (chunk: ArrayBuffer) => void;
  onError?: (error: unknown) => void;
};

export type AudioStreamController = {
  start: () => Promise<void>;
  stop: () => void;
  isRecording: () => boolean;
};

// 16kHz mono PCM16 little-endian framing
const TARGET_SAMPLE_RATE = 16000;
const FRAME_DURATION_MS = 250; // ~250ms per frame

export function createMicrophonePcmStream(callbacks: AudioStreamCallbacks): AudioStreamController {
  let mediaStream: MediaStream | null = null;
  let audioContext: AudioContext | null = null;
  let processor: ScriptProcessorNode | null = null;
  let sourceNode: MediaStreamAudioSourceNode | null = null;
  let running = false;

  // Buffer for resampling and framing
  let resampleBuffer: Float32Array = new Float32Array(0);
  let lastEmitTime = 0;

  async function start() {
    if (running) return;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
      audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 });
      sourceNode = audioContext.createMediaStreamSource(mediaStream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);

      sourceNode.connect(processor);
      processor.connect(audioContext.destination);

      running = true;
      lastEmitTime = performance.now();

      processor.onaudioprocess = (event: AudioProcessingEvent) => {
        if (!running) return;
        const input = event.inputBuffer.getChannelData(0);
        // Resample from context sample rate to 16kHz
        const resampled = resampleLinear(input, audioContext!.sampleRate, TARGET_SAMPLE_RATE);
        // Append to buffer
        const concat = new Float32Array(resampleBuffer.length + resampled.length);
        concat.set(resampleBuffer, 0);
        concat.set(resampled, resampleBuffer.length);
        resampleBuffer = concat;

        const now = performance.now();
        if (now - lastEmitTime >= FRAME_DURATION_MS) {
          const samplesPerFrame = Math.round((TARGET_SAMPLE_RATE * FRAME_DURATION_MS) / 1000);
          const framesToEmit = Math.floor(resampleBuffer.length / samplesPerFrame);
          if (framesToEmit > 0) {
            const emitCount = framesToEmit * samplesPerFrame;
            const frame = resampleBuffer.slice(0, emitCount);
            resampleBuffer = resampleBuffer.slice(emitCount);
            lastEmitTime = now;
            const pcm = floatTo16BitPCM(frame);
            try {
              callbacks.onChunk(pcm.buffer);
            } catch (err) {
              callbacks.onError?.(err);
            }
          }
        }
      };
    } catch (err) {
      callbacks.onError?.(err);
      stop();
    }
  }

  function stop() {
    running = false;
    if (processor) {
      processor.disconnect();
      processor.onaudioprocess = null;
      processor = null;
    }
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    resampleBuffer = new Float32Array(0);
  }

  return {
    start,
    stop,
    isRecording: () => running,
  };
}

function resampleLinear(input: Float32Array, inputRate: number, targetRate: number): Float32Array {
  if (inputRate === targetRate) return input;
  const ratio = inputRate / targetRate;
  const newLength = Math.floor(input.length / ratio);
  const output = new Float32Array(newLength);
  let pos = 0;
  for (let i = 0; i < newLength; i++) {
    const idx = i * ratio;
    const idxLow = Math.floor(idx);
    const idxHigh = Math.min(idxLow + 1, input.length - 1);
    const weight = idx - idxLow;
    output[i] = input[idxLow] * (1 - weight) + input[idxHigh] * weight;
  }
  return output;
}

function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}
