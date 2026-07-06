# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare one-shot transcription with realtime chunked transcription."""

import argparse
import asyncio
import json
import os
import unicodedata
from difflib import SequenceMatcher
from typing import Sequence

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


def normalize_text(
    text: str,
    *,
    ignore_case: bool,
    strip_marks: bool,
    remove_punctuation: bool,
) -> str:
    text = unicodedata.normalize("NFKC", text)
    if ignore_case:
        text = text.casefold()

    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if strip_marks and category.startswith("M"):
            continue
        if remove_punctuation and category.startswith("P"):
            continue
        chars.append(char)

    return " ".join("".join(chars).split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def char_tokens(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


def word_tokens(text: str) -> list[str]:
    return text.split()


def calculate_metrics(
    baseline: str,
    realtime: str,
    *,
    ignore_case: bool,
    strip_marks: bool,
    remove_punctuation: bool,
) -> dict[str, float | int]:
    baseline_norm = normalize_text(
        baseline,
        ignore_case=ignore_case,
        strip_marks=strip_marks,
        remove_punctuation=remove_punctuation,
    )
    realtime_norm = normalize_text(
        realtime,
        ignore_case=ignore_case,
        strip_marks=strip_marks,
        remove_punctuation=remove_punctuation,
    )

    baseline_chars = char_tokens(baseline_norm)
    realtime_chars = char_tokens(realtime_norm)
    baseline_words = word_tokens(baseline_norm)
    realtime_words = word_tokens(realtime_norm)

    cer = error_rate(baseline_chars, realtime_chars)
    wer = error_rate(baseline_words, realtime_words)
    return {
        "baseline_normalized_chars": len(baseline_chars),
        "realtime_normalized_chars": len(realtime_chars),
        "baseline_normalized_words": len(baseline_words),
        "realtime_normalized_words": len(realtime_words),
        "normalized_similarity": SequenceMatcher(
            None, baseline_norm, realtime_norm
        ).ratio(),
        "cer": cer,
        "cer_similarity": max(0.0, 1.0 - cer),
        "wer": wer,
        "wer_similarity": max(0.0, 1.0 - wer),
    }


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

    metrics = calculate_metrics(
        baseline,
        realtime,
        ignore_case=not args.keep_case,
        strip_marks=not args.keep_diacritics,
        remove_punctuation=not args.keep_punctuation,
    )
    print("\n=== One-shot baseline ===")
    print(baseline)
    print("\n=== Realtime chunked ===")
    print(realtime)
    print("\n=== Summary ===")
    print(f"baseline_chars={len(baseline)}")
    print(f"realtime_chars={len(realtime)}")
    print(
        "normalization="
        f"ignore_case={not args.keep_case}, "
        f"strip_diacritics={not args.keep_diacritics}, "
        f"remove_punctuation={not args.keep_punctuation}"
    )
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}={value:.4f}")
        else:
            print(f"{name}={value}")


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
    parser.add_argument(
        "--keep-case",
        action="store_true",
        help="Keep case differences when calculating metrics.",
    )
    parser.add_argument(
        "--keep-diacritics",
        action="store_true",
        help="Keep combining marks/diacritics when calculating metrics.",
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Keep punctuation when calculating metrics.",
    )
    args = parser.parse_args()
    asyncio.run(compare(args))


if __name__ == "__main__":
    main()
