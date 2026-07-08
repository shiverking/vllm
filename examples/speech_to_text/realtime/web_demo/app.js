/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 100;
const CONNECT_TIMEOUT_MS = 5000;

const serverUrlInput = document.querySelector("#server-url");
const modelInput = document.querySelector("#model");
const startButton = document.querySelector("#start-btn");
const stopButton = document.querySelector("#stop-btn");
const transcript = document.querySelector("#transcript");
const statusBadge = document.querySelector("#status");

let socket = null;
let mediaStream = null;
let audioContext = null;
let sourceNode = null;
let processorNode = null;
let pendingInput = new Float32Array(0);
let committedFinalAudio = false;
let sessionConfigured = false;
let sessionReady = false;
let running = false;

const text = {
  idle: "\u672a\u8fde\u63a5",
  emptyUrl: "\u5730\u5740\u4e3a\u7a7a",
  connecting: "\u8fde\u63a5\u4e2d",
  recording: "\u6b63\u5728\u5f55\u97f3",
  transcribing: "\u6b63\u5728\u8f6c\u5199",
  done: "\u5df2\u5b8c\u6210",
  error: "\u53d1\u751f\u9519\u8bef",
  disconnected: "\u8fde\u63a5\u5df2\u65ad\u5f00",
  connectionError: "\u8fde\u63a5\u9519\u8bef",
  startupFailed: "\u542f\u52a8\u5931\u8d25",
  waiting: "\u7b49\u5f85\u7ed3\u679c",
  ended: "\u5df2\u7ed3\u675f",
  serverError: "\u670d\u52a1\u7aef\u8fd4\u56de\u9519\u8bef",
  connectFailed: "\u65e0\u6cd5\u8fde\u63a5\u5230 realtime \u670d\u52a1",
  connectTimeout: "\u8fde\u63a5 realtime \u670d\u52a1\u8d85\u65f6",
  errorPrefix: "\u9519\u8bef",
};

function setStatus(value, kind = "idle") {
  statusBadge.textContent = value;
  statusBadge.classList.toggle("live", kind === "live");
  statusBadge.classList.toggle("error", kind === "error");
}

function setControls(isRunning) {
  startButton.disabled = isRunning;
  stopButton.disabled = !isRunning;
  serverUrlInput.disabled = isRunning;
  modelInput.disabled = isRunning;
}

function setWaitingControls() {
  startButton.disabled = true;
  stopButton.disabled = true;
  serverUrlInput.disabled = true;
  modelInput.disabled = true;
}

function appendTranscript(value) {
  transcript.value += value;
  transcript.scrollTop = transcript.scrollHeight;
}

function setFinalTranscript(value) {
  transcript.value = value || transcript.value;
  transcript.scrollTop = transcript.scrollHeight;
}

function encodeBase64(bytes) {
  let binary = "";
  const blockSize = 0x8000;
  for (let i = 0; i < bytes.length; i += blockSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + blockSize));
  }
  return btoa(binary);
}

function mergeFloat32(left, right) {
  if (!left.length) {
    return right;
  }
  const merged = new Float32Array(left.length + right.length);
  merged.set(left);
  merged.set(right, left.length);
  return merged;
}

function downsampleTo16k(input, inputSampleRate) {
  if (inputSampleRate === TARGET_SAMPLE_RATE) {
    return input;
  }

  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j += 1) {
      sum += input[j];
      count += 1;
    }
    output[i] = count > 0 ? sum / count : 0;
  }

  return output;
}

function floatToPcm16Bytes(input) {
  const bytes = new Uint8Array(input.length * 2);
  const view = new DataView(bytes.buffer);

  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    const value = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(i * 2, value, true);
  }

  return bytes;
}

function sendEvent(event) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }
  socket.send(JSON.stringify(event));
}

function configureSession(force = false) {
  if (sessionConfigured && !force) {
    return;
  }

  const model = modelInput.value.trim();
  if (model) {
    sendEvent({ type: "session.update", model });
  }
  sendEvent({ type: "input_audio_buffer.commit" });
  sessionConfigured = true;
  sessionReady = true;
  setStatus(text.recording, "live");
}

function sendAudioChunk(samples) {
  const pcmBytes = floatToPcm16Bytes(samples);
  sendEvent({
    type: "input_audio_buffer.append",
    audio: encodeBase64(pcmBytes),
  });
}

function flushPendingAudio() {
  if (!pendingInput.length) {
    return;
  }
  sendAudioChunk(pendingInput);
  pendingInput = new Float32Array(0);
}

function queueAudio(inputBuffer) {
  if (
    !running ||
    !sessionReady ||
    !socket ||
    socket.readyState !== WebSocket.OPEN
  ) {
    return;
  }

  const downsampled = downsampleTo16k(inputBuffer, audioContext.sampleRate);
  pendingInput = mergeFloat32(pendingInput, downsampled);

  const chunkSize = Math.floor((TARGET_SAMPLE_RATE * CHUNK_DURATION_MS) / 1000);
  while (pendingInput.length >= chunkSize) {
    const chunk = pendingInput.slice(0, chunkSize);
    pendingInput = pendingInput.slice(chunkSize);
    sendAudioChunk(chunk);
  }
}

function handleServerMessage(message) {
  let event;
  try {
    event = JSON.parse(message.data);
  } catch (error) {
    console.warn("Ignoring non-JSON realtime message:", message.data);
    return;
  }

  if (event.type === "session.created") {
    configureSession(true);
    return;
  }

  if (event.type === "transcription.delta") {
    appendTranscript(event.delta || "");
    setStatus(text.transcribing, "live");
    return;
  }

  if (event.type === "transcription.done") {
    setFinalTranscript(event.text);
    setStatus(text.done, "idle");
    closeSocket();
    setControls(false);
    return;
  }

  if (event.type === "error") {
    const detail = event.error?.message || event.error || text.serverError;
    setStatus(text.error, "error");
    appendTranscript(`\n[${text.errorPrefix}] ${detail}\n`);
    stopRecording(false);
  }
}

function connectWebSocket(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const timeout = window.setTimeout(() => {
      ws.close();
      reject(new Error(text.connectTimeout));
    }, CONNECT_TIMEOUT_MS);

    ws.addEventListener(
      "open",
      () => {
        window.clearTimeout(timeout);
        resolve(ws);
      },
      { once: true },
    );
    ws.addEventListener(
      "error",
      () => {
        window.clearTimeout(timeout);
        reject(new Error(text.connectFailed));
      },
      { once: true },
    );
  });
}

async function startAudioCapture() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContext();
  sourceNode = audioContext.createMediaStreamSource(mediaStream);

  processorNode = audioContext.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    queueAudio(new Float32Array(input));
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);
}

async function startRecording() {
  const url = serverUrlInput.value.trim();
  if (!url) {
    setStatus(text.emptyUrl, "error");
    return;
  }

  transcript.value = "";
  pendingInput = new Float32Array(0);
  committedFinalAudio = false;
  sessionConfigured = false;
  sessionReady = false;
  running = true;
  setControls(true);
  setStatus(text.connecting, "live");

  try {
    socket = await connectWebSocket(url);
    socket.addEventListener("message", handleServerMessage);
    socket.addEventListener("close", () => {
      if (running) {
        setStatus(text.disconnected, "error");
        stopRecording(false);
      }
    });
    socket.addEventListener("error", () => {
      if (running) {
        setStatus(text.connectionError, "error");
        stopRecording(false);
      }
    });

    configureSession();
    await startAudioCapture();
  } catch (error) {
    appendTranscript(`[${text.errorPrefix}] ${error.message}\n`);
    setStatus(text.startupFailed, "error");
    stopRecording(false);
  }
}

function stopTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function stopAudioNodes() {
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
}

async function closeAudioContext() {
  if (audioContext) {
    await audioContext.close();
    audioContext = null;
  }
}

function closeSocket() {
  if (socket) {
    const currentSocket = socket;
    socket = null;
    if (
      currentSocket.readyState === WebSocket.OPEN ||
      currentSocket.readyState === WebSocket.CONNECTING
    ) {
      currentSocket.close();
    }
  }
}

async function stopRecording(sendFinalCommit = true) {
  const wasRunning = running;
  running = false;
  sessionReady = false;

  stopTracks();
  stopAudioNodes();

  if (
    sendFinalCommit &&
    wasRunning &&
    socket &&
    socket.readyState === WebSocket.OPEN &&
    !committedFinalAudio
  ) {
    flushPendingAudio();
    sendEvent({ type: "input_audio_buffer.commit", final: true });
    committedFinalAudio = true;
    setWaitingControls();
    setStatus(text.waiting, "live");
  } else if (wasRunning) {
    closeSocket();
    setControls(false);
    setStatus(text.ended, "idle");
  } else {
    closeSocket();
    setControls(false);
  }

  await closeAudioContext();
}

startButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", () => stopRecording(true));

window.addEventListener("beforeunload", () => {
  stopTracks();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }
});
