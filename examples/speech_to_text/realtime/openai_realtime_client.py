# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM Realtime WebSocket API to perform
audio transcription by uploading an audio file.

Before running this script, you must start the vLLM server with a realtime-capable
model, for example:

    vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --enforce-eager

Requirements:
- vllm with audio support
- websockets
- numpy

The script:
1. Connects to the Realtime WebSocket endpoint
2. Converts an audio file to PCM16 @ 16kHz
3. Sends audio chunks to the server
4. Receives and prints transcription as it streams
"""

import argparse
import asyncio
import json
import time

import numpy as np
import pybase64 as base64
import websockets

from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def audio_to_pcm16_bytes(audio_path: str) -> bytes:
    """
    Load an audio file and convert it to PCM16 @ 16kHz.
    """
    # Load audio and resample to 16kHz mono
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    # Convert to PCM16
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


async def realtime_transcribe(
    audio_path: str,
    host: str,
    port: int,
    model: str,
    chunk_duration_ms: int,
    realtime_pacing: bool,
    print_timestamps: bool,
):
    """
    Connect to the Realtime API and transcribe an audio file.
    """
    uri = f"ws://{host}:{port}/v1/realtime"

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print(f"Session created: {response['id']}")
        else:
            print(f"Unexpected response: {response}")
            return

        # Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Signal ready to start
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Convert audio file to PCM16
        print(f"Loading audio from: {audio_path}")
        audio_bytes = audio_to_pcm16_bytes(audio_path)

        # Send audio in time-sized chunks. 100ms at 16kHz PCM16 mono = 3200B.
        chunk_size = max(
            1,
            int(SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_duration_ms / 1000),
        )
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
        audio_duration_s = len(audio_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

        async def send_audio() -> None:
            print(
                f"Sending {total_chunks} audio chunks "
                f"({chunk_duration_ms}ms each, {audio_duration_s:.2f}s audio)..."
            )
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i : i + chunk_size]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("utf-8"),
                        }
                    )
                )
                if realtime_pacing:
                    await asyncio.sleep(chunk_duration_ms / 1000)

            # Signal all audio is sent
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )
            print("Audio sent. Waiting for remaining transcription...\n")

        async def receive_transcription() -> None:
            start_time = time.perf_counter()
            print("Transcription: ", end="", flush=True)
            while True:
                response = json.loads(await ws.recv())
                if response["type"] == "transcription.delta":
                    if print_timestamps:
                        elapsed = time.perf_counter() - start_time
                        print(f"\n[{elapsed:7.3f}s] ", end="", flush=True)
                    print(response["delta"], end="", flush=True)
                elif response["type"] == "transcription.done":
                    print(f"\n\nFinal transcription: {response['text']}")
                    if response.get("usage"):
                        print(f"Usage: {response['usage']}")
                    break
                elif response["type"] == "error":
                    print(f"\nError: {response['error']}")
                    break

        await asyncio.gather(send_audio(), receive_transcription())


def main(args):
    if args.audio_path:
        audio_path = args.audio_path
    else:
        # Use default audio asset
        audio_path = str(AudioAsset("mary_had_lamb").get_local_path())
        print(f"No audio path provided, using default: {audio_path}")

    asyncio.run(
        realtime_transcribe(
            audio_path,
            args.host,
            args.port,
            args.model,
            args.chunk_duration_ms,
            args.realtime_pacing,
            args.print_timestamps,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realtime WebSocket Transcription Client"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="Model that is served and should be pinged.",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=100,
        help="Audio duration per websocket chunk in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--realtime-pacing",
        action="store_true",
        help="Sleep between chunks to simulate microphone capture timing.",
    )
    parser.add_argument(
        "--print-timestamps",
        action="store_true",
        help="Print elapsed time before each transcription delta.",
    )
    args = parser.parse_args()
    main(args)
