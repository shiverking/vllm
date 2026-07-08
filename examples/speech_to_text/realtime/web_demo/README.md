# Realtime ASR Web Demo

This is a minimal browser demo for the vLLM Realtime WebSocket API. The browser
only connects to a local Python backend. The Python backend serves the page,
connects to the model server, relays microphone audio, can capture system
playback audio, and relays streaming transcription events back to the page.

Install the demo dependencies:

```powershell
uv pip install websockets soundcard numpy
```

Start the backend:

```powershell
cd examples\speech_to_text\realtime\web_demo
..\..\..\..\.venv\Scripts\python realtime_web_demo_server.py --target ws://135.30.72.115:1025/v1/realtime
```

Open the page printed by the backend:

```text
http://127.0.0.1:8080
```

The page defaults to:

```text
Model server URL: ws://135.30.72.115:1025/v1/realtime
Model: Qwen3-ASR-1.7B
```

Use **Start** to capture microphone audio from the browser.

Use **Capture System Audio** to capture computer playback audio in the Python
backend. On Windows this uses WASAPI loopback through the `soundcard` package.

The TTFT indicator records the elapsed time from clicking a capture button to
receiving the first non-empty `transcription.delta` event, in milliseconds.

The browser sends microphone audio to the Python backend at:

```text
ws://127.0.0.1:8765/ws
```

The browser starts system audio capture through:

```text
ws://127.0.0.1:8765/system-audio
```

The backend relays realtime events to the model server:

```json
{"type": "session.update", "model": "<model>"}
{"type": "input_audio_buffer.commit"}
{"type": "input_audio_buffer.append", "audio": "<base64 pcm16>"}
{"type": "input_audio_buffer.commit", "final": true}
```

The page displays `transcription.delta` events as streaming text and replaces
the text area with `transcription.done.text` when the final result arrives.
