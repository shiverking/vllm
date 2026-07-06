# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare one-shot transcription with realtime chunked transcription."""

import argparse
import asyncio
import json
import os
from difflib import SequenceMatcher

import httpx
import numpy as np
import pybase64 as base64
import websockets

from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def audio_to_pcm16_bytes(audio_path: str) -> bytes:
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


async def transcribe_once(
    audio_path: str,
    *,
    host: str,
    port: int,
    model: str,
    language: str | None,
) -> str:
    url = f"http://{host}:{port}/v1/audio/transcriptions"
    data = {
        "model": model,
        "response_format": "json",
        "temperature": "0.0",
    }
    if language:
        data["language"] = language

    async with httpx.AsyncClient(timeout=None) as client:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (
                    os.path.basename(audio_path),
                    audio_file,
                    "application/octet-stream",
                )
            }
            response = await client.post(url, data=data, files=files)
            response.raise_for_status()
    payload = response.json()
    return str(payload.get("text", ""))


async def transcribe_realtime(
    audio_path: str,
    *,
    host: str,
    port: int,
    model: str,
    chunk_duration_ms: int,
    realtime_pacing: bool,
    print_deltas: bool,
) -> str:
    uri = f"ws://{host}:{port}/v1/realtime"
    audio_bytes = audio_to_pcm16_bytes(audio_path)
    chunk_size = max(
        1,
        int(SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_duration_ms / 1000),
    )

    async with websockets.connect(uri) as ws:
        response = json.loads(await ws.recv())
        if response["type"] != "session.created":
            raise RuntimeError(f"Unexpected realtime response: {response}")

        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def send_audio() -> None:
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
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )

        async def receive_text() -> str:
            pieces: list[str] = []
            while True:
                response = json.loads(await ws.recv())
                if response["type"] == "transcription.delta":
                    delta = response["delta"]
                    pieces.append(delta)
                    if print_deltas:
                        print(delta, end="", flush=True)
                elif response["type"] == "transcription.done":
                    return str(response["text"])
                elif response["type"] == "error":
                    raise RuntimeError(f"Realtime error: {response['error']}")

        _, realtime_text = await asyncio.gather(send_audio(), receive_text())
        return realtime_text


def normalized_similarity(left: str, right: str) -> float:
    left_norm = " ".join(left.split())
    right_norm = " ".join(right.split())
    return SequenceMatcher(None, left_norm, right_norm).ratio()


async def compare(args: argparse.Namespace) -> None:
    audio_path = args.audio_path
    if audio_path is None:
        audio_path = str(AudioAsset("mary_had_lamb").get_local_path())
        print(f"No audio path provided, using default: {audio_path}")

    print("Running one-shot transcription baseline...")
    baseline = await transcribe_once(
        audio_path,
        host=args.host,
        port=args.port,
        model=args.model,
        language=args.language,
    )

    print("Running realtime chunked transcription...")
    if args.print_deltas:
        print("Realtime deltas: ", end="", flush=True)
    realtime = await transcribe_realtime(
        audio_path,
        host=args.host,
        port=args.port,
        model=args.model,
        chunk_duration_ms=args.chunk_duration_ms,
        realtime_pacing=args.realtime_pacing,
        print_deltas=args.print_deltas,
    )
    if args.print_deltas:
        print()

    similarity = normalized_similarity(baseline, realtime)
    print("\n=== One-shot baseline ===")
    print(baseline)
    print("\n=== Realtime chunked ===")
    print(realtime)
    print("\n=== Summary ===")
    print(f"baseline_chars={len(baseline)}")
    print(f"realtime_chars={len(realtime)}")
    print(f"normalized_similarity={similarity:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one-shot and realtime ASR outputs."
    )
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint for the one-shot transcription endpoint.",
    )
    parser.add_argument("--chunk-duration-ms", type=int, default=100)
    parser.add_argument("--realtime-pacing", action="store_true")
    parser.add_argument("--print-deltas", action="store_true")
    args = parser.parse_args()
    asyncio.run(compare(args))


if __name__ == "__main__":
    main()
