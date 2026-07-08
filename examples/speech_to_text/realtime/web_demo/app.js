/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 100;

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
let sessionReady = false;
let running = false;

function setStatus(text, kind = "idle") {
  statusBadge.textContent = text;
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

function appendTranscript(text) {
  transcript.value += text;
  transcript.scrollTop = transcript.scrollHeight;
}

function setFinalTranscript(text) {
  transcript.value = text || transcript.value;
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
    const model = modelInput.value.trim();
    if (model) {
      sendEvent({ type: "session.update", model });
    }
    sendEvent({ type: "input_audio_buffer.commit" });
    sessionReady = true;
    setStatus("正在录音", "live");
    return;
  }

  if (event.type === "transcription.delta") {
    appendTranscript(event.delta || "");
    setStatus("正在转写", "live");
    return;
  }

  if (event.type === "transcription.done") {
    setFinalTranscript(event.text);
    setStatus("已完成", "idle");
    closeSocket();
    setControls(false);
    return;
  }

  if (event.type === "error") {
    const detail = event.error?.message || event.error || "服务端返回错误";
    setStatus("发生错误", "error");
    appendTranscript(`\n[错误] ${detail}\n`);
    stopRecording(false);
  }
}

function connectWebSocket(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);

    ws.addEventListener("open", () => resolve(ws), { once: true });
    ws.addEventListener(
      "error",
      () => reject(new Error("无法连接到 realtime 服务")),
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
    setStatus("地址为空", "error");
    return;
  }

  transcript.value = "";
  pendingInput = new Float32Array(0);
  committedFinalAudio = false;
  sessionReady = false;
  running = true;
  setControls(true);
  setStatus("连接中", "live");

  try {
    socket = await connectWebSocket(url);
    socket.addEventListener("message", handleServerMessage);
    socket.addEventListener("close", () => {
      if (running) {
        setStatus("连接已断开", "error");
        stopRecording(false);
      }
    });
    socket.addEventListener("error", () => {
      if (running) {
        setStatus("连接错误", "error");
        stopRecording(false);
      }
    });

    await startAudioCapture();
  } catch (error) {
    appendTranscript(`[错误] ${error.message}\n`);
    setStatus("启动失败", "error");
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
    setStatus("等待结果", "live");
  } else if (wasRunning) {
    closeSocket();
    setControls(false);
    setStatus("已结束", "idle");
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
