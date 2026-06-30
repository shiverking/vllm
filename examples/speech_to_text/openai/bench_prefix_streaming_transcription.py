# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent benchmark for REST prefix-streaming audio transcription.

This benchmark exercises the caller-side "pseudo streaming" flow implemented by
``openai_prefix_streaming_transcription_client.py``:

1. Split each audio file into time steps.
2. Send a sliding audio window to the OpenAI-compatible transcription endpoint.
3. Feed stable text back through ``response_prefix``.
4. Optionally use SSE streaming to measure first-delta TTFT.

Example:

    python examples/speech_to_text/openai/bench_prefix_streaming_transcription.py \
        --audio-dir /workspace/asr_mini_real_world/mini_real_world_120/audio \
        --api-base http://localhost:1025/v1/audio \
        --concurrency 4 \
        --num-requests 20 \
        --chunk-seconds 10 \
        --max-audio-window-seconds 30 \
        --stream
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from vllm.multimodal.media.audio import load_audio

try:
    import openai_prefix_streaming_transcription_client as prefix_client
except ImportError:
    from examples.speech_to_text.openai import (
        openai_prefix_streaming_transcription_client as prefix_client,
    )

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".webm"}


def find_all_audio(audio_dir: str) -> list[Path]:
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        raise RuntimeError(f"no audio files found in {audio_dir}")
    return files


def get_audio_duration_ms(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                rate = wav_file.getframerate()
                if rate > 0:
                    return wav_file.getnframes() / rate * 1000.0
        except Exception:
            pass

    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio_path))
        if info.samplerate > 0:
            return info.frames / info.samplerate * 1000.0
    except Exception:
        return None

    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def fmt_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def print_metric(label: str, value: float | None, precision: int = 2) -> None:
    print(f"{label:<44}{fmt_float(value, precision):>12}")


def print_int_metric(label: str, value: int) -> None:
    print(f"{label:<44}{value:>12}")


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def run_one_prefix_streaming_request(
    *,
    audio_path: Path,
    api_base: str,
    model: str,
    task: str,
    language: str | None,
    to_language: str | None,
    sample_rate: int,
    chunk_seconds: float,
    max_audio_window_seconds: float,
    max_prefix_words: int,
    holdback_words: int,
    stream: bool,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    request_start = time.perf_counter()
    audio, actual_sample_rate = load_audio(str(audio_path), sr=sample_rate, mono=True)
    chunk_samples = max(1, int(chunk_seconds * actual_sample_rate))
    max_window_samples = (
        int(max_audio_window_seconds * actual_sample_rate)
        if max_audio_window_seconds > 0
        else None
    )
    endpoint = "translations" if task == "translate" else "transcriptions"

    state = prefix_client.PrefixStreamingState()
    request_latencies_ms: list[float] = []
    round_ttft_ms: list[float] = []
    total_uploaded_audio_ms = 0.0
    total_uploaded_bytes = 0
    first_round_ttft_ms: float | None = None
    rounds = 0

    for end in range(chunk_samples, len(audio) + chunk_samples, chunk_samples):
        rounds += 1
        end = min(end, len(audio))
        start = 0
        if max_window_samples is not None:
            start = max(0, end - max_window_samples)

        prefix = prefix_client.cap_text_by_words(
            state.stable_text,
            max_prefix_words,
        )
        result = prefix_client.post_audio_request(
            api_base=api_base,
            endpoint=endpoint,
            audio=audio[start:end],
            sample_rate=actual_sample_rate,
            model=model,
            language=language,
            to_language=to_language,
            response_prefix=prefix,
            stream=stream,
            temperature=temperature,
            timeout=timeout,
        )
        request_latencies_ms.append(result.latency_ms)
        total_uploaded_audio_ms += result.upload_audio_ms
        total_uploaded_bytes += result.upload_bytes
        if result.ttft_ms is not None:
            round_ttft_ms.append(result.ttft_ms)
            if first_round_ttft_ms is None:
                first_round_ttft_ms = result.ttft_ms

        candidate_text = prefix_client.merge_history_and_candidate(
            state.stable_text,
            prefix_client.merge_prefix_and_response(prefix, result.text),
        )
        state.stable_text, state.unstable_text = prefix_client.split_with_holdback(
            candidate_text,
            holdback_words,
        )

        if end >= len(audio):
            break

    final_prefix = prefix_client.cap_text_by_words(
        state.stable_text,
        max_prefix_words,
    )
    final_audio = audio
    if max_window_samples is not None:
        final_audio = audio[-max_window_samples:]

    final_result = prefix_client.post_audio_request(
        api_base=api_base,
        endpoint=endpoint,
        audio=final_audio,
        sample_rate=actual_sample_rate,
        model=model,
        language=language,
        to_language=to_language,
        response_prefix=final_prefix,
        stream=stream,
        temperature=temperature,
        timeout=timeout,
    )
    request_latencies_ms.append(final_result.latency_ms)
    total_uploaded_audio_ms += final_result.upload_audio_ms
    total_uploaded_bytes += final_result.upload_bytes
    if final_result.ttft_ms is not None:
        round_ttft_ms.append(final_result.ttft_ms)
        if first_round_ttft_ms is None:
            first_round_ttft_ms = final_result.ttft_ms

    state.stable_text = prefix_client.merge_history_and_candidate(
        state.stable_text,
        prefix_client.merge_prefix_and_response(final_prefix, final_result.text),
    )
    state.unstable_text = ""

    e2e_ms = (time.perf_counter() - request_start) * 1000.0
    audio_duration_ms = len(audio) / actual_sample_rate * 1000.0
    rtf = e2e_ms / audio_duration_ms if audio_duration_ms > 0 else None

    if first_round_ttft_ms is None and state.stable_text:
        first_round_ttft_ms = request_latencies_ms[0]

    return {
        "ok": True,
        "file": audio_path.name,
        "audio_path": str(audio_path.resolve()),
        "audio_duration_ms": audio_duration_ms,
        "e2e_ms": e2e_ms,
        "ttft_ms": first_round_ttft_ms,
        "round_ttft_ms": round_ttft_ms,
        "request_latencies_ms": request_latencies_ms,
        "avg_round_latency_ms": mean(request_latencies_ms),
        "max_round_latency_ms": max(request_latencies_ms),
        "rounds": rounds,
        "http_requests": len(request_latencies_ms),
        "uploaded_audio_ms": total_uploaded_audio_ms,
        "uploaded_bytes": total_uploaded_bytes,
        "rtf": rtf,
        "final_text_words": len(state.stable_text.split()),
        "error": None,
    }


def run_worker(
    request_id: int,
    audio_path: Path,
    args: argparse.Namespace,
    start_event: threading.Event,
) -> dict[str, Any]:
    start_event.wait()
    try:
        result = run_one_prefix_streaming_request(
            audio_path=audio_path,
            api_base=args.api_base,
            model=args.model,
            task=args.task,
            language=args.language,
            to_language=args.to_language,
            sample_rate=args.sample_rate,
            chunk_seconds=args.chunk_seconds,
            max_audio_window_seconds=args.max_audio_window_seconds,
            max_prefix_words=args.max_prefix_words,
            holdback_words=args.holdback_words,
            stream=args.stream,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        result["request_id"] = request_id
        return result
    except Exception as exc:
        return {
            "ok": False,
            "request_id": request_id,
            "file": audio_path.name,
            "audio_path": str(audio_path.resolve()),
            "error": str(exc),
        }


def perform_warmup(args: argparse.Namespace) -> list[float]:
    if args.no_warmup or args.warmup_file is None:
        return []

    warmup_path = Path(args.warmup_file)
    if not warmup_path.exists():
        print(f"WARNING: warmup file not found: {args.warmup_file}")
        return []

    print("\n" + "=" * 80)
    print(f"WARMUP: {args.warmup_iterations} request(s), file={warmup_path}")
    print("=" * 80)
    times: list[float] = []
    for i in range(args.warmup_iterations):
        start = time.perf_counter()
        try:
            result = run_one_prefix_streaming_request(
                audio_path=warmup_path,
                api_base=args.api_base,
                model=args.model,
                task=args.task,
                language=args.language,
                to_language=args.to_language,
                sample_rate=args.sample_rate,
                chunk_seconds=args.chunk_seconds,
                max_audio_window_seconds=args.max_audio_window_seconds,
                max_prefix_words=args.max_prefix_words,
                holdback_words=args.holdback_words,
                stream=args.stream,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            times.append(result["e2e_ms"])
            print(
                f"  OK warmup {i + 1}: "
                f"e2e_ms={result['e2e_ms']:.2f}, "
                f"ttft_ms={fmt_float(result['ttft_ms'])}, "
                f"http_requests={result['http_requests']}"
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"  FAIL warmup {i + 1}: elapsed_ms={elapsed_ms:.2f}, {exc}")

    print("=" * 80)
    print("Warmup complete. Starting benchmark...\n")
    return times


def print_benchmark_result(
    *,
    success_results: list[dict[str, Any]],
    failed_results: list[dict[str, Any]],
    concurrency: int,
    benchmark_duration_ms: float,
) -> None:
    e2e_values = [r["e2e_ms"] for r in success_results]
    ttft_values = [r["ttft_ms"] for r in success_results if r["ttft_ms"] is not None]
    rtf_values = [r["rtf"] for r in success_results if r["rtf"] is not None]
    round_latency_values = [
        value for r in success_results for value in r["request_latencies_ms"]
    ]
    uploaded_audio_ms = sum(r["uploaded_audio_ms"] for r in success_results)
    source_audio_ms = sum(r["audio_duration_ms"] for r in success_results)
    total_http_requests = sum(r["http_requests"] for r in success_results)
    duration_s = benchmark_duration_ms / 1000.0

    request_throughput = safe_div(len(success_results), duration_s)
    audio_throughput = safe_div(source_audio_ms, benchmark_duration_ms)
    http_request_rate = safe_div(total_http_requests, duration_s)

    print("=" * 56)
    print("        Prefix Streaming ASR Benchmark Result       ")
    print("=" * 56)
    print_int_metric("Successful logical requests:", len(success_results))
    print_int_metric("Failed logical requests:", len(failed_results))
    print_int_metric("Maximum logical concurrency:", concurrency)
    print_int_metric("Total HTTP audio requests:", total_http_requests)
    print_metric("Benchmark duration (ms):", benchmark_duration_ms, 2)
    print_metric("Logical request throughput (req/s):", request_throughput, 2)
    print_metric("HTTP request throughput (req/s):", http_request_rate, 2)
    print("-" * 56)
    print("                  End-to-End Latency                ")
    print("-" * 56)
    for key, value in summarize(e2e_values).items():
        print_metric(f"{key.upper()} E2E (ms):", value, 2)
    print("-" * 56)
    print("              First Round Time to First Text        ")
    print("-" * 56)
    for key, value in summarize(ttft_values).items():
        print_metric(f"{key.upper()} TTFT (ms):", value, 2)
    print("-" * 56)
    print("                Per HTTP Request Latency            ")
    print("-" * 56)
    for key, value in summarize(round_latency_values).items():
        print_metric(f"{key.upper()} round latency (ms):", value, 2)
    print("-" * 56)
    print("                       Audio                         ")
    print("-" * 56)
    print_metric("Source audio total (ms):", source_audio_ms, 2)
    print_metric("Uploaded audio total (ms):", uploaded_audio_ms, 2)
    print_metric(
        "Upload amplification:",
        safe_div(uploaded_audio_ms, source_audio_ms),
        3,
    )
    print_metric("Audio throughput (audio ms / wall ms):", audio_throughput, 3)
    print_metric("Mean RTF:", mean(rtf_values), 3)
    print_metric("P99 RTF:", percentile(rtf_values, 99), 3)
    print("=" * 56)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent benchmark for REST prefix-streaming ASR."
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--to-language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--max-audio-window-seconds", type=float, default=30.0)
    parser.add_argument("--max-prefix-words", type=int, default=100)
    parser.add_argument("--holdback-words", type=int, default=5)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--output-file", default="prefix_streaming_benchmark.json")
    parser.add_argument("--warmup-file", default=None)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")

    audio_paths = find_all_audio(args.audio_dir)
    if args.warmup_file is not None:
        warmup_path = Path(args.warmup_file).resolve()
        audio_paths = [p for p in audio_paths if p.resolve() != warmup_path]
    if not audio_paths:
        raise RuntimeError("no benchmark audio files remain")

    num_requests = args.num_requests or len(audio_paths)
    tasks = [
        audio_paths[idx % len(audio_paths)].resolve()
        for idx in range(num_requests)
    ]

    print(f"Found {len(audio_paths)} benchmark audio files in {args.audio_dir}")
    print(f"Benchmark logical requests: {num_requests}")
    print(f"Maximum logical concurrency: {args.concurrency}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(
        "Pseudo streaming: "
        f"chunk_ms={args.chunk_seconds * 1000:.0f}, "
        f"max_audio_window_ms={args.max_audio_window_seconds * 1000:.0f}, "
        f"max_prefix_words={args.max_prefix_words}, "
        f"holdback_words={args.holdback_words}, stream={args.stream}"
    )

    warmup_times = perform_warmup(args)
    start_event = threading.Event()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_worker, idx + 1, audio_path, args, start_event)
            for idx, audio_path in enumerate(tasks)
        ]

        benchmark_start = time.perf_counter()
        start_event.set()

        iterator = as_completed(futures)
        if tqdm is not None:
            pbar = tqdm(total=len(futures), desc="Benchmark", unit="req")
            try:
                for future in iterator:
                    result = future.result()
                    results.append(result)
                    postfix = {
                        "success": sum(1 for r in results if r["ok"]),
                        "failed": sum(1 for r in results if not r["ok"]),
                        "file": result["file"],
                    }
                    if result["ok"]:
                        postfix["e2e_ms"] = f"{result['e2e_ms']:.2f}"
                        postfix["ttft_ms"] = fmt_float(result["ttft_ms"])
                    pbar.set_postfix(postfix)
                    pbar.update(1)
            finally:
                pbar.close()
        else:
            for completed, future in enumerate(iterator, start=1):
                result = future.result()
                results.append(result)
                print(f"Progress: {completed}/{len(futures)} requests completed")

        benchmark_duration_ms = (time.perf_counter() - benchmark_start) * 1000.0

    success_results = [r for r in results if r["ok"]]
    failed_results = [r for r in results if not r["ok"]]
    print_benchmark_result(
        success_results=success_results,
        failed_results=failed_results,
        concurrency=args.concurrency,
        benchmark_duration_ms=benchmark_duration_ms,
    )

    output = {
        "config": vars(args),
        "warmup": {
            "performed": bool(warmup_times),
            "iterations": len(warmup_times),
            "e2e_ms": warmup_times,
            "avg_e2e_ms": mean(warmup_times),
        },
        "summary": {
            "successful_requests": len(success_results),
            "failed_requests": len(failed_results),
            "benchmark_duration_ms": benchmark_duration_ms,
            "e2e_ms": summarize([r["e2e_ms"] for r in success_results]),
            "ttft_ms": summarize(
                [r["ttft_ms"] for r in success_results if r["ttft_ms"] is not None]
            ),
            "round_latency_ms": summarize(
                [
                    value
                    for result in success_results
                    for value in result["request_latencies_ms"]
                ]
            ),
            "rtf": summarize(
                [r["rtf"] for r in success_results if r["rtf"] is not None]
            ),
        },
        "per_request": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
