# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare streaming transcription with realtime chunked transcription."""

import argparse
import asyncio
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
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


@dataclass
class TranscriptionResult:
    text: str
    ttft_s: float
    e2e_s: float


def clean_streamed_asr_text(text: str) -> str:
    if not text:
        return ""
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[1]
    elif re.fullmatch(r"language(?:\s+[A-Za-z]+)?", text):
        return ""
    text = re.sub(r"(^|\s)language\s+[A-Za-z]+(?=[A-Z])", r"\1", text)
    return re.sub(r"(^|\s)language(?=\S)", r"\1", text)


def audio_to_pcm16_bytes(audio_path: str) -> bytes:
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


def get_audio_duration_s(audio_path: str) -> float:
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    return len(audio) / SAMPLE_RATE


async def transcribe_stream(
    audio_path: str,
    *,
    host: str,
    port: int,
    model: str,
    language: str | None,
    max_completion_tokens: int,
    print_deltas: bool,
) -> TranscriptionResult:
    url = f"http://{host}:{port}/v1/audio/transcriptions"
    data = {
        "model": model,
        "response_format": "json",
        "stream": "true",
        "temperature": "0.0",
        "max_completion_tokens": str(max_completion_tokens),
    }
    if language:
        data["language"] = language

    raw_pieces: list[str] = []
    emitted_text = ""
    ttft_s: float | None = None
    async with httpx.AsyncClient(timeout=None) as client:
        with open(audio_path, "rb") as audio_file:
            files = {
                "file": (
                    os.path.basename(audio_path),
                    audio_file,
                    "application/octet-stream",
                )
            }
            start_time = time.perf_counter()
            async with client.stream("POST", url, data=data, files=files) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_line = line[len("data: ") :]
                    if data_line == "[DONE]":
                        break
                    payload = json.loads(data_line)
                    if "error" in payload:
                        raise RuntimeError(f"Transcription stream error: {payload}")
                    choices = payload.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        raw_pieces.append(delta)
                        cleaned_text = clean_streamed_asr_text("".join(raw_pieces))
                        if cleaned_text.startswith(emitted_text):
                            cleaned_delta = cleaned_text[len(emitted_text) :]
                        else:
                            cleaned_delta = cleaned_text
                        if cleaned_delta:
                            if ttft_s is None:
                                ttft_s = time.perf_counter() - start_time
                            emitted_text = cleaned_text
                            if print_deltas:
                                print(cleaned_delta, end="", flush=True)
            e2e_s = time.perf_counter() - start_time

    return TranscriptionResult(
        text=clean_streamed_asr_text("".join(raw_pieces)),
        ttft_s=ttft_s if ttft_s is not None else e2e_s,
        e2e_s=e2e_s,
    )


async def transcribe_realtime(
    audio_path: str,
    *,
    host: str,
    port: int,
    model: str,
    chunk_duration_ms: int,
    realtime_pacing: bool,
    print_deltas: bool,
) -> TranscriptionResult:
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
        timing_start_time: float | None = None
        timing_started = asyncio.Event()

        async def send_audio() -> None:
            nonlocal timing_start_time
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
            timing_start_time = time.perf_counter()
            timing_started.set()

        async def receive_text() -> TranscriptionResult:
            raw_pieces: list[str] = []
            emitted_text = ""
            first_delta_time: float | None = None
            while True:
                response = json.loads(await ws.recv())
                if response["type"] == "transcription.delta":
                    delta = response["delta"]
                    raw_pieces.append(delta)
                    cleaned_text = clean_streamed_asr_text("".join(raw_pieces))
                    if cleaned_text.startswith(emitted_text):
                        cleaned_delta = cleaned_text[len(emitted_text) :]
                    else:
                        cleaned_delta = cleaned_text
                    if cleaned_delta:
                        if first_delta_time is None:
                            first_delta_time = time.perf_counter()
                        emitted_text = cleaned_text
                        if print_deltas:
                            print(cleaned_delta, end="", flush=True)
                elif response["type"] == "transcription.done":
                    done_time = time.perf_counter()
                    await timing_started.wait()
                    assert timing_start_time is not None
                    e2e_s = max(0.0, done_time - timing_start_time)
                    if first_delta_time is None:
                        ttft_s = e2e_s
                    else:
                        ttft_s = max(0.0, first_delta_time - timing_start_time)
                    return TranscriptionResult(
                        text=clean_streamed_asr_text(str(response["text"])),
                        ttft_s=ttft_s,
                        e2e_s=e2e_s,
                    )
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
    reference: str,
    hypothesis: str,
    *,
    ignore_case: bool,
    strip_marks: bool,
    remove_punctuation: bool,
) -> dict[str, float | int]:
    reference_norm = normalize_text(
        reference,
        ignore_case=ignore_case,
        strip_marks=strip_marks,
        remove_punctuation=remove_punctuation,
    )
    hypothesis_norm = normalize_text(
        hypothesis,
        ignore_case=ignore_case,
        strip_marks=strip_marks,
        remove_punctuation=remove_punctuation,
    )

    reference_chars = char_tokens(reference_norm)
    hypothesis_chars = char_tokens(hypothesis_norm)
    reference_words = word_tokens(reference_norm)
    hypothesis_words = word_tokens(hypothesis_norm)

    cer = error_rate(reference_chars, hypothesis_chars)
    wer = error_rate(reference_words, hypothesis_words)
    return {
        "reference_normalized_chars": len(reference_chars),
        "hypothesis_normalized_chars": len(hypothesis_chars),
        "reference_normalized_words": len(reference_words),
        "hypothesis_normalized_words": len(hypothesis_words),
        "normalized_similarity": SequenceMatcher(
            None, reference_norm, hypothesis_norm
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

    audio_duration_s = get_audio_duration_s(audio_path)

    print("Running /v1/audio/transcriptions stream=true baseline...")
    if args.print_deltas:
        print("Transcription stream baseline deltas: ", end="", flush=True)
    baseline = await transcribe_stream(
        audio_path,
        host=args.host,
        port=args.port,
        model=args.model,
        language=args.language,
        max_completion_tokens=args.max_completion_tokens,
        print_deltas=args.print_deltas,
    )
    if args.print_deltas:
        print()

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

    realtime_metrics = calculate_metrics(
        baseline.text,
        realtime.text,
        ignore_case=not args.keep_case,
        strip_marks=not args.keep_diacritics,
        remove_punctuation=not args.keep_punctuation,
    )
    print("\n=== Transcription Stream Baseline ===")
    print(baseline.text)
    print("\n=== Realtime chunked ===")
    print(realtime.text)
    print("\n=== Summary ===")
    print(f"audio_duration={audio_duration_s * 1000:.3f}ms")
    print(f"transcription_stream_max_completion_tokens={args.max_completion_tokens}")
    print("realtime_timing_origin=audio_send_done")
    print(f"baseline_chars={len(baseline.text)}")
    print(f"realtime_chars={len(realtime.text)}")
    print_performance("transcription_stream_baseline", baseline, audio_duration_s)
    print_performance("realtime", realtime, audio_duration_s)
    print(
        "normalization="
        f"ignore_case={not args.keep_case}, "
        f"strip_diacritics={not args.keep_diacritics}, "
        f"remove_punctuation={not args.keep_punctuation}"
    )
    print_metrics("realtime", realtime_metrics)


def print_performance(
    name: str,
    result: TranscriptionResult,
    audio_duration_s: float,
) -> None:
    e2e_rtf = result.e2e_s / audio_duration_s if audio_duration_s else float("inf")
    print(f"{name}_ttft={result.ttft_s * 1000:.3f}ms")
    print(f"{name}_e2e={result.e2e_s * 1000:.3f}ms")
    print(f"{name}_rtf={e2e_rtf:.3f}")


def print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}_{metric_name}={value:.4f}")
        else:
            print(f"{name}_{metric_name}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare transcription stream and realtime ASR outputs."
    )
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint for the transcription stream endpoint.",
    )
    parser.add_argument("--chunk-duration-ms", type=int, default=100)
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for the transcription stream baseline.",
    )
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
