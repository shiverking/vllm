# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-audio transcription with offline VAD segmented transcription."""

import argparse
import asyncio
import io
import json
import os
import re
import time
import unicodedata
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

import httpx
import numpy as np

from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio
from vllm.multimodal.realtime_vad import (
    RealtimeVADSegmenter,
    SileroSpeechDetector,
)

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


@dataclass
class TranscriptionResult:
    text: str
    ttft_s: float
    e2e_s: float


@dataclass
class SegmentResult:
    index: int
    duration_s: float
    emit_offset_s: float
    request_start_offset_s: float
    result: TranscriptionResult


@dataclass
class AudioSegment:
    index: int
    audio: np.ndarray
    emit_offset_s: float


@dataclass
class PipelineResult:
    transcription: TranscriptionResult
    segments: list[SegmentResult]
    vad_init_s: float
    vad_time_s: float
    first_segment_emit_s: float
    first_request_start_s: float
    first_request_ttft_s: float
    first_token_segment_index: int


def clean_streamed_asr_text(text: str) -> str:
    if not text:
        return ""
    if "<asr_text>" in text:
        text = text.rsplit("<asr_text>", 1)[1]
    elif re.fullmatch(r"language(?:\s+[A-Za-z]+)?", text):
        return ""
    text = re.sub(r"(^|\s)language\s+[A-Za-z]+(?=[A-Z])", r"\1", text)
    return re.sub(r"(^|\s)language(?=\S)", r"\1", text)


def load_audio_16k(audio_path: str) -> np.ndarray:
    audio, _ = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(BYTES_PER_SAMPLE)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()


async def transcribe_stream_bytes(
    client: httpx.AsyncClient,
    audio_bytes: bytes,
    *,
    filename: str,
    host: str,
    port: int,
    model: str,
    language: str | None,
    max_completion_tokens: int,
    timing_start_time: float | None = None,
    print_deltas: bool = False,
    content_type: str = "application/octet-stream",
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
    start_time = timing_start_time or time.perf_counter()
    files = {"file": (filename, audio_bytes, content_type)}
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
                if ttft_s is None:
                    ttft_s = time.perf_counter() - start_time
                raw_pieces.append(delta)
                cleaned_text = clean_streamed_asr_text("".join(raw_pieces))
                if cleaned_text.startswith(emitted_text):
                    cleaned_delta = cleaned_text[len(emitted_text) :]
                else:
                    cleaned_delta = cleaned_text
                if cleaned_delta:
                    emitted_text = cleaned_text
                    if print_deltas:
                        print(cleaned_delta, end="", flush=True)
    e2e_s = time.perf_counter() - start_time
    return TranscriptionResult(
        text=clean_streamed_asr_text("".join(raw_pieces)),
        ttft_s=ttft_s if ttft_s is not None else e2e_s,
        e2e_s=e2e_s,
    )


async def transcribe_vad_pipeline(
    audio: np.ndarray,
    *,
    host: str,
    port: int,
    model: str,
    language: str | None,
    max_completion_tokens: int,
    concurrency: int,
    print_segment_text: bool,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
    max_segment_s: float,
    vad_chunk_duration_ms: int,
    vad_onnx: bool,
) -> PipelineResult:
    vad_init_start_time = time.perf_counter()
    detector = SileroSpeechDetector(
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        onnx=vad_onnx,
    )
    segmenter = RealtimeVADSegmenter(
        detector,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        max_segment_duration_s=max_segment_s,
    )
    vad_init_s = time.perf_counter() - vad_init_start_time
    pipeline_start_time = time.perf_counter()
    queue: asyncio.Queue[AudioSegment | None] = asyncio.Queue(
        maxsize=max(1, concurrency * 2)
    )
    segment_results: list[SegmentResult] = []
    vad_time_s: float | None = None

    async def produce_segments() -> None:
        nonlocal vad_time_s
        chunk_samples = max(1, int(SAMPLE_RATE * vad_chunk_duration_ms / 1000))
        next_index = 1
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            segments = await asyncio.to_thread(segmenter.write_audio, chunk)
            for segment in segments:
                emit_offset_s = time.perf_counter() - pipeline_start_time
                print(
                    f"vad_segment_{next_index}: "
                    f"{len(segment) / SAMPLE_RATE:.3f}s "
                    f"({len(segment)} samples), "
                    f"emit={emit_offset_s * 1000:.3f}ms."
                )
                await queue.put(AudioSegment(next_index, segment, emit_offset_s))
                next_index += 1
            await asyncio.sleep(0)

        remaining = await asyncio.to_thread(segmenter.flush)
        if remaining is not None and len(remaining) > 0:
            emit_offset_s = time.perf_counter() - pipeline_start_time
            print(
                f"vad_segment_{next_index}: "
                f"{len(remaining) / SAMPLE_RATE:.3f}s "
                f"({len(remaining)} samples), "
                f"emit={emit_offset_s * 1000:.3f}ms."
            )
            await queue.put(AudioSegment(next_index, remaining, emit_offset_s))

        vad_time_s = time.perf_counter() - pipeline_start_time
        for _ in range(concurrency):
            await queue.put(None)

    async with httpx.AsyncClient(timeout=None) as client:

        async def consume_segments() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    request_start_time = time.perf_counter()
                    request_start_offset_s = (
                        request_start_time - pipeline_start_time
                    )
                    result = await transcribe_stream_bytes(
                        client,
                        audio_to_wav_bytes(item.audio),
                        filename=f"vad_segment_{item.index}.wav",
                        host=host,
                        port=port,
                        model=model,
                        language=language,
                        max_completion_tokens=max_completion_tokens,
                        timing_start_time=request_start_time,
                        content_type="audio/wav",
                    )
                    duration_s = len(item.audio) / SAMPLE_RATE
                    if print_segment_text:
                        print(
                            f"vad_segment_{item.index}_duration="
                            f"{duration_s * 1000:.3f}ms"
                        )
                        print(f"vad_segment_{item.index}_text={result.text}")
                    segment_results.append(
                        SegmentResult(
                            item.index,
                            duration_s,
                            item.emit_offset_s,
                            request_start_offset_s,
                            result,
                        )
                    )
                finally:
                    queue.task_done()

        producer_task = asyncio.create_task(produce_segments())
        consumer_tasks = [
            asyncio.create_task(consume_segments()) for _ in range(concurrency)
        ]
        try:
            await producer_task
            await asyncio.gather(*consumer_tasks)
        finally:
            for task in consumer_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*consumer_tasks, return_exceptions=True)

    if not segment_results:
        raise RuntimeError("Silero VAD did not produce any speech segments.")

    segment_results.sort(key=lambda item: item.index)
    full_text = " ".join(
        item.result.text.strip()
        for item in segment_results
        if item.result.text.strip()
    )
    first_token_result = min(
        segment_results,
        key=lambda item: item.request_start_offset_s + item.result.ttft_s,
    )
    first_ttft_s = (
        first_token_result.request_start_offset_s
        + first_token_result.result.ttft_s
    )
    e2e_s = max(
        item.request_start_offset_s + item.result.e2e_s
        for item in segment_results
    )
    return PipelineResult(
        transcription=TranscriptionResult(
            text=full_text,
            ttft_s=first_ttft_s,
            e2e_s=e2e_s,
        ),
        segments=segment_results,
        vad_init_s=vad_init_s,
        vad_time_s=vad_time_s if vad_time_s is not None else e2e_s,
        first_segment_emit_s=segment_results[0].emit_offset_s,
        first_request_start_s=min(
            item.request_start_offset_s for item in segment_results
        ),
        first_request_ttft_s=first_token_result.result.ttft_s,
        first_token_segment_index=first_token_result.index,
    )


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
        "ref_chars": len(reference_chars),
        "hyp_chars": len(hypothesis_chars),
        "ref_words": len(reference_words),
        "hyp_words": len(hypothesis_words),
        "similarity": SequenceMatcher(
            None, reference_norm, hypothesis_norm
        ).ratio(),
        "cer": cer,
        "cer_sim": max(0.0, 1.0 - cer),
        "wer": wer,
        "wer_sim": max(0.0, 1.0 - wer),
    }


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


async def compare(args: argparse.Namespace) -> None:
    audio_path = args.audio_path
    if audio_path is None:
        audio_path = str(AudioAsset("mary_had_lamb").get_local_path())
        print(f"No audio path provided, using default: {audio_path}")

    audio = load_audio_16k(audio_path)
    audio_duration_s = len(audio) / SAMPLE_RATE

    print("Running /v1/audio/transcriptions stream=true baseline...")
    async with httpx.AsyncClient(timeout=None) as client:
        with open(audio_path, "rb") as audio_file:
            baseline = await transcribe_stream_bytes(
                client,
                audio_file.read(),
                filename=os.path.basename(audio_path),
                host=args.host,
                port=args.port,
                model=args.model,
                language=args.language,
                max_completion_tokens=args.max_completion_tokens,
                print_deltas=args.print_deltas,
            )
    if args.print_deltas:
        print()

    print("Running pipelined Silero VAD + transcription streams...")
    pipeline_result = await transcribe_vad_pipeline(
        audio,
        host=args.host,
        port=args.port,
        model=args.model,
        language=args.language,
        max_completion_tokens=args.segment_max_completion_tokens,
        concurrency=args.segment_concurrency,
        print_segment_text=args.print_segment_text,
        threshold=args.vad_threshold,
        min_speech_ms=args.vad_min_speech_ms,
        min_silence_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        max_segment_s=args.vad_max_segment_s,
        vad_chunk_duration_ms=args.vad_chunk_duration_ms,
        vad_onnx=args.vad_onnx,
    )
    vad_transcription = pipeline_result.transcription
    segment_results = pipeline_result.segments

    metrics = calculate_metrics(
        baseline.text,
        vad_transcription.text,
        ignore_case=not args.keep_case,
        strip_marks=not args.keep_diacritics,
        remove_punctuation=not args.keep_punctuation,
    )

    speech_duration_s = sum(item.duration_s for item in segment_results)
    print("\n=== Transcription Stream Baseline ===")
    print(baseline.text)
    print("\n=== Offline VAD Transcription ===")
    print(vad_transcription.text)
    print("\n=== Summary ===")
    print("\n[audio]")
    print(f"audio_duration={audio_duration_s * 1000:.3f}ms")
    print(f"vad_init={pipeline_result.vad_init_s * 1000:.3f}ms")
    print(f"vad_time={pipeline_result.vad_time_s * 1000:.3f}ms")
    print(f"vad_segments={len(segment_results)}")
    print(f"vad_speech_duration={speech_duration_s * 1000:.3f}ms")
    print(
        "vad_speech_ratio="
        f"{speech_duration_s / audio_duration_s if audio_duration_s else 0.0:.4f}"
    )

    print("\n[config]")
    print(f"baseline_max_tokens={args.max_completion_tokens}")
    print(f"vad_segment_max_tokens={args.segment_max_completion_tokens}")
    print(f"vad_concurrency={args.segment_concurrency}")
    print(f"vad_onnx={args.vad_onnx}")

    print("\n[text]")
    print(f"baseline_chars={len(baseline.text)}")
    print(f"vad_chars={len(vad_transcription.text)}")

    print("\n[performance]")
    print_performance("baseline", baseline, audio_duration_s)
    print_performance("vad", vad_transcription, audio_duration_s)
    print(f"vad_first_segment_emit={pipeline_result.first_segment_emit_s * 1000:.3f}ms")
    print(
        "vad_first_request_start="
        f"{pipeline_result.first_request_start_s * 1000:.3f}ms"
    )
    print(f"vad_first_request_ttft={pipeline_result.first_request_ttft_s * 1000:.3f}ms")
    print(f"vad_first_token_segment={pipeline_result.first_token_segment_index}")

    print("\n[quality]")
    print(
        "normalization="
        f"ignore_case={not args.keep_case}, "
        f"strip_diacritics={not args.keep_diacritics}, "
        f"remove_punctuation={not args.keep_punctuation}"
    )
    print_metrics("vad", metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-audio transcription stream with offline Silero VAD "
            "segmented transcription streams."
        )
    )
    parser.add_argument("--audio_path", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint for transcription requests.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1536,
        help="Maximum tokens for the full-audio transcription baseline.",
    )
    parser.add_argument(
        "--segment-max-completion-tokens",
        type=int,
        default=256,
        help="Maximum tokens for each offline VAD segment request.",
    )
    parser.add_argument(
        "--segment-concurrency",
        type=int,
        default=4,
        help="Maximum concurrent transcription requests for VAD segments.",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=700)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300)
    parser.add_argument("--vad-max-segment-s", type=float, default=25.0)
    parser.add_argument("--vad-chunk-duration-ms", type=int, default=1000)
    parser.add_argument(
        "--vad-onnx",
        action="store_true",
        help="Use Silero VAD ONNXRuntime backend instead of the JIT backend.",
    )
    parser.add_argument("--print-deltas", action="store_true")
    parser.add_argument("--print-segment-text", action="store_true")
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
    if args.segment_concurrency <= 0:
        raise ValueError("--segment-concurrency must be positive.")
    asyncio.run(compare(args))


if __name__ == "__main__":
    main()
