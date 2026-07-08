# Realtime ASR Web Demo

This is a minimal browser demo for the vLLM Realtime WebSocket API. It captures
microphone audio, sends PCM16 16 kHz mono chunks to the server, and streams
transcription text back into the page.

Start a realtime-capable vLLM server first, for example:

```powershell
vllm serve Qwen3-ASR-1.7B --enforce-eager
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
ws://135.30.72.115:1025/v1/realtime
```

and model:

```text
Qwen3-ASR-1.7B
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
