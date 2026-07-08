# Realtime ASR Web Demo

This is a minimal browser demo for the vLLM Realtime WebSocket API. It captures
microphone audio, sends PCM16 16 kHz mono chunks to the server, and streams
transcription text back into the page.

Start a realtime-capable vLLM server first, for example:

```powershell
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager
```

Then serve this directory from localhost:

```powershell
cd examples\speech_to_text\realtime\web_demo
..\..\..\..\.venv\Scripts\python -m http.server 8080
```

Open:

```text
http://localhost:8080
```

The page defaults to:

```text
ws://127.0.0.1:8000/v1/realtime
```

Change the WebSocket URL and model name in the page before clicking Start.

The demo sends:

```json
{"type": "session.update", "model": "<model>"}
{"type": "input_audio_buffer.commit"}
{"type": "input_audio_buffer.append", "audio": "<base64 pcm16>"}
{"type": "input_audio_buffer.commit", "final": true}
```

It displays `transcription.delta` events as streaming text and replaces the
text area with `transcription.done.text` when the final result arrives.
