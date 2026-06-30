# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative prefix-streaming benchmark for Arabic and Thai ASR.

Arabic and Thai can be more likely to trigger very long repeated generation
when caller-side prefix streaming feeds model output back through
``response_prefix``. This benchmark keeps the same pseudo-streaming shape as
``bench_prefix_streaming_transcription.py`` but adds protective limits:

* per-request ``max_completion_tokens``
* prefix caps by both words and characters
* holdback by both words and characters
* response length and repetition checks
* wall-clock stream cutoff
* bad-round isolation so abnormal text is not used as the next prefix
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

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
SUPPORTED_LANGUAGES = {"ar", "th"}


@dataclass
class SafeRequestResult:
    text: str
    latency_ms: float
    ttft_ms: float | None
    upload_audio_ms: float
    upload_bytes: int
    timed_out: bool = False
    truncated: bool = False
    repeated: bool = False
    parse_errors: int = 0


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def fmt_float(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def print_metric(label: str, value: float | None, precision: int = 2) -> None:
    print(f"{label:<46}{fmt_float(value, precision):>12}")


def print_int_metric(label: str, value: int) -> None:
    print(f"{label:<46}{value:>12}")


def find_all_audio(audio_dir: str, language: str) -> list[Path]:
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if language in SUPPORTED_LANGUAGES:
        files = [p for p in files if infer_language(p, language) == language]
    else:
        files = [
            p
            for p in files
            if infer_optional_language(p, language) in SUPPORTED_LANGUAGES
        ]

    if not files:
        raise RuntimeError(f"no Arabic/Thai audio files found in {audio_dir}")
    return files


def infer_optional_language(audio_path: Path, configured_language: str) -> str | None:
    try:
        return infer_language(audio_path, configured_language)
    except ValueError:
        return None


def infer_language(audio_path: Path, configured_language: str) -> str:
    if configured_language in SUPPORTED_LANGUAGES:
        return configured_language

    name = audio_path.stem.lower()
    if name.startswith("ar") or "arabic" in name:
        return "ar"
    if name.startswith("th") or "thai" in name:
        return "th"
    raise ValueError(
        f"cannot infer Arabic/Thai language from file name: {audio_path.name}. "
        "Use --language ar or --language th."
    )


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


def cap_text_by_words_and_chars(
    text: str,
    max_words: int,
    max_chars: int,
) -> str:
    capped = prefix_client.cap_text_by_words(text, max_words)
    if max_chars > 0 and len(capped) > max_chars:
        capped = capped[-max_chars:]
        first_space = capped.find(" ")
        if first_space >= 0:
            capped = capped[first_space + 1 :]
    return capped.strip()


def split_with_dual_holdback(
    text: str,
    holdback_words: int,
    holdback_chars: int,
) -> tuple[str, str]:
    stable_by_words, unstable_by_words = prefix_client.split_with_holdback(
        text,
        holdback_words,
    )
    if holdback_chars <= 0 or len(text) <= holdback_chars:
        return stable_by_words, unstable_by_words

    stable_by_chars = text[:-holdback_chars]
    unstable_by_chars = text[-holdback_chars:]
    stable_len = min(len(stable_by_words), len(stable_by_chars))
    return text[:stable_len].rstrip(), text[stable_len:]


def has_repeated_char_ngram(
    text: str,
    ngram_size: int,
    threshold: int,
) -> bool:
    if ngram_size <= 0 or threshold <= 0:
        return False

    compact = "".join(text.split())
    if len(compact) < ngram_size:
        return False

    counts: Counter[str] = Counter()
    for idx in range(0, len(compact) - ngram_size + 1):
        ngram = compact[idx : idx + ngram_size]
        counts[ngram] += 1
        if counts[ngram] > threshold:
            return True
    return False


def build_drop_reasons(
    result: SafeRequestResult,
    *,
    max_response_words: int,
    max_response_chars: int,
) -> list[str]:
    reasons: list[str] = []
    response_words = len(result.text.split())
    response_chars = len(result.text)

    if result.timed_out:
        reasons.append("timeout")
    if result.truncated:
        reasons.append("truncated")
    if result.repeated:
        reasons.append("repeated")
    if max_response_words > 0 and response_words > max_response_words:
        reasons.append("too_many_words")
    if max_response_chars > 0 and response_chars > max_response_chars:
        reasons.append("too_many_chars")
    return reasons


def post_safe_audio_request(
    *,
    api_base: str,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    model: str,
    language: str,
    to_language: str | None,
    response_prefix: str,
    stream: bool,
    temperature: float,
    timeout: float,
    max_completion_tokens: int,
    max_request_ms: float,
    max_response_chars: int,
    repeat_ngram_size: int,
    repeat_ngram_threshold: int,
) -> SafeRequestResult:
    data: dict[str, Any] = {
        "model": model,
        "response_format": "json",
        "response_prefix": response_prefix,
        "stream": "true" if stream else "false",
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "language": language,
    }
    if to_language:
        data["to_language"] = to_language

    request_start = time.perf_counter()
    deadline = request_start + max_request_ms / 1000.0 if max_request_ms > 0 else None
    ttft_ms: float | None = None
    timed_out = False
    truncated = False
    parse_errors = 0

    with prefix_client.audio_to_wav_buffer(audio, sample_rate) as wav_buffer:
        upload_bytes = wav_buffer.getbuffer().nbytes
        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        with requests.post(
            prefix_client.build_audio_endpoint_url(api_base, endpoint),
            data=data,
            files=files,
            stream=stream,
            timeout=timeout,
        ) as response:
            response.raise_for_status()

            if stream:
                text_parts: list[str] = []
                response_chars = 0
                for line in response.iter_lines(decode_unicode=True):
                    now = time.perf_counter()
                    if deadline is not None and now > deadline:
                        timed_out = True
                        response.close()
                        break
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    if line.strip() == "[DONE]":
                        break

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue

                    delta = prefix_client.extract_stream_delta(payload)
                    if not delta:
                        continue
                    if ttft_ms is None:
                        ttft_ms = seconds_to_ms(now - request_start)
                    text_parts.append(delta)
                    response_chars += len(delta)
                    if max_response_chars > 0 and response_chars > max_response_chars:
                        truncated = True
                        response.close()
                        break
                text = "".join(text_parts)
            else:
                text = response.json()["text"]

    latency_ms = seconds_to_ms(time.perf_counter() - request_start)
    if deadline is not None and latency_ms > max_request_ms:
        timed_out = True
    if max_response_chars > 0 and len(text) > max_response_chars:
        text = text[:max_response_chars]
        truncated = True

    repeated = has_repeated_char_ngram(
        text,
        repeat_ngram_size,
        repeat_ngram_threshold,
    )
    return SafeRequestResult(
        text=text,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        upload_audio_ms=seconds_to_ms(len(audio) / sample_rate),
        upload_bytes=upload_bytes,
        timed_out=timed_out,
        truncated=truncated,
        repeated=repeated,
        parse_errors=parse_errors,
    )


def update_state_from_result(
    state: prefix_client.PrefixStreamingState,
    prefix: str,
    result_text: str,
    holdback_words: int,
    holdback_chars: int,
) -> None:
    candidate_text = prefix_client.merge_history_and_candidate(
        state.stable_text,
        prefix_client.merge_prefix_and_response(prefix, result_text),
    )
    state.stable_text, state.unstable_text = split_with_dual_holdback(
        candidate_text,
        holdback_words,
        holdback_chars,
    )


def run_one_request(
    *,
    request_id: int | None,
    audio_path: Path,
    args: argparse.Namespace,
    language_override: str | None = None,
) -> dict[str, Any]:
    request_start = time.perf_counter()
    language = language_override or infer_language(audio_path, args.language)
    audio, sample_rate = load_audio(str(audio_path), sr=args.sample_rate, mono=True)
    chunk_samples = max(1, int(args.chunk_seconds * sample_rate))
    max_window_samples = (
        int(args.max_audio_window_seconds * sample_rate)
        if args.max_audio_window_seconds > 0
        else None
    )
    endpoint = "translations" if args.task == "translate" else "transcriptions"

    state = prefix_client.PrefixStreamingState()
    request_latencies_ms: list[float] = []
    round_ttft_ms: list[float] = []
    first_round_ttft_ms: float | None = None
    total_uploaded_audio_ms = 0.0
    total_uploaded_bytes = 0
    rounds = 0
    bad_rounds = 0
    dropped_rounds = 0
    timeout_rounds = 0
    repeated_rounds = 0
    too_long_rounds = 0
    consecutive_bad_rounds = 0
    drop_reasons_all: list[str] = []

    for end in range(chunk_samples, len(audio) + chunk_samples, chunk_samples):
        rounds += 1
        end = min(end, len(audio))
        start = 0
        if max_window_samples is not None:
            start = max(0, end - max_window_samples)

        prefix = cap_text_by_words_and_chars(
            state.stable_text,
            args.max_prefix_words,
            args.max_prefix_chars,
        )
        if args.log_rounds:
            print(
                "[round-start] "
                f"request_id={request_id} file={audio_path.name} "
                f"language={language} round={rounds} "
                f"window_ms={start / sample_rate * 1000.0:.0f}-"
                f"{end / sample_rate * 1000.0:.0f} "
                f"prefix_words={len(prefix.split())} "
                f"prefix_chars={len(prefix)}",
                flush=True,
            )

        result = post_safe_audio_request(
            api_base=args.api_base,
            endpoint=endpoint,
            audio=audio[start:end],
            sample_rate=sample_rate,
            model=args.model,
            language=language,
            to_language=args.to_language,
            response_prefix=prefix,
            stream=args.stream,
            temperature=args.temperature,
            timeout=args.timeout,
            max_completion_tokens=args.max_tokens,
            max_request_ms=args.max_request_ms,
            max_response_chars=args.max_response_chars,
            repeat_ngram_size=args.repeat_ngram_size,
            repeat_ngram_threshold=args.repeat_ngram_threshold,
        )
        request_latencies_ms.append(result.latency_ms)
        total_uploaded_audio_ms += result.upload_audio_ms
        total_uploaded_bytes += result.upload_bytes
        if result.ttft_ms is not None:
            round_ttft_ms.append(result.ttft_ms)
            if first_round_ttft_ms is None:
                first_round_ttft_ms = result.ttft_ms

        drop_reasons = build_drop_reasons(
            result,
            max_response_words=args.max_response_words,
            max_response_chars=args.max_response_chars,
        )
        if drop_reasons:
            bad_rounds += 1
            dropped_rounds += 1
            consecutive_bad_rounds += 1
            drop_reasons_all.extend(drop_reasons)
            if result.timed_out:
                timeout_rounds += 1
            if result.repeated:
                repeated_rounds += 1
            if (
                "too_many_words" in drop_reasons
                or "too_many_chars" in drop_reasons
                or "truncated" in drop_reasons
            ):
                too_long_rounds += 1
        else:
            consecutive_bad_rounds = 0
            update_state_from_result(
                state,
                prefix,
                result.text,
                args.holdback_words,
                args.holdback_chars,
            )

        if args.log_rounds:
            status = "drop" if drop_reasons else "ok"
            print(
                "[round-done] "
                f"request_id={request_id} file={audio_path.name} "
                f"language={language} round={rounds} "
                f"latency_ms={result.latency_ms:.0f} "
                f"ttft_ms={result.ttft_ms if result.ttft_ms is not None else -1:.0f} "
                f"response_words={len(result.text.split())} "
                f"response_chars={len(result.text)} status={status} "
                f"drop_reason={','.join(drop_reasons) or '-'}",
                flush=True,
            )

        if consecutive_bad_rounds >= args.max_consecutive_bad_rounds:
            e2e_ms = seconds_to_ms(time.perf_counter() - request_start)
            return build_request_result(
                ok=False,
                status="failed",
                request_id=request_id,
                audio_path=audio_path,
                language=language,
                audio=audio,
                sample_rate=sample_rate,
                e2e_ms=e2e_ms,
                first_round_ttft_ms=first_round_ttft_ms,
                round_ttft_ms=round_ttft_ms,
                request_latencies_ms=request_latencies_ms,
                rounds=rounds,
                total_uploaded_audio_ms=total_uploaded_audio_ms,
                total_uploaded_bytes=total_uploaded_bytes,
                final_text=state.stable_text,
                bad_rounds=bad_rounds,
                dropped_rounds=dropped_rounds,
                timeout_rounds=timeout_rounds,
                repeated_rounds=repeated_rounds,
                too_long_rounds=too_long_rounds,
                drop_reasons_all=drop_reasons_all,
                error="max consecutive bad rounds reached",
            )

        if end >= len(audio):
            break

    final_prefix = cap_text_by_words_and_chars(
        state.stable_text,
        args.max_prefix_words,
        args.max_prefix_chars,
    )
    final_audio = audio
    if max_window_samples is not None:
        final_audio = audio[-max_window_samples:]

    if args.log_rounds:
        print(
            "[final-start] "
            f"request_id={request_id} file={audio_path.name} language={language} "
            f"prefix_words={len(final_prefix.split())} "
            f"prefix_chars={len(final_prefix)}",
            flush=True,
        )

    final_result = post_safe_audio_request(
        api_base=args.api_base,
        endpoint=endpoint,
        audio=final_audio,
        sample_rate=sample_rate,
        model=args.model,
        language=language,
        to_language=args.to_language,
        response_prefix=final_prefix,
        stream=args.stream,
        temperature=args.temperature,
        timeout=args.timeout,
        max_completion_tokens=args.final_max_tokens,
        max_request_ms=args.max_request_ms,
        max_response_chars=args.max_response_chars,
        repeat_ngram_size=args.repeat_ngram_size,
        repeat_ngram_threshold=args.repeat_ngram_threshold,
    )
    request_latencies_ms.append(final_result.latency_ms)
    total_uploaded_audio_ms += final_result.upload_audio_ms
    total_uploaded_bytes += final_result.upload_bytes
    if final_result.ttft_ms is not None:
        round_ttft_ms.append(final_result.ttft_ms)
        if first_round_ttft_ms is None:
            first_round_ttft_ms = final_result.ttft_ms

    final_drop_reasons = build_drop_reasons(
        final_result,
        max_response_words=args.max_response_words,
        max_response_chars=args.max_response_chars,
    )
    if final_drop_reasons:
        bad_rounds += 1
        dropped_rounds += 1
        drop_reasons_all.extend(final_drop_reasons)
        if final_result.timed_out:
            timeout_rounds += 1
        if final_result.repeated:
            repeated_rounds += 1
        if (
            "too_many_words" in final_drop_reasons
            or "too_many_chars" in final_drop_reasons
            or "truncated" in final_drop_reasons
        ):
            too_long_rounds += 1
    else:
        state.stable_text = prefix_client.merge_history_and_candidate(
            state.stable_text,
            prefix_client.merge_prefix_and_response(final_prefix, final_result.text),
        )
        state.unstable_text = ""

    if args.log_rounds:
        status = "drop" if final_drop_reasons else "ok"
        print(
            "[final-done] "
            f"request_id={request_id} file={audio_path.name} "
            f"language={language} latency_ms={final_result.latency_ms:.0f} "
            "ttft_ms="
            f"{final_result.ttft_ms if final_result.ttft_ms is not None else -1:.0f} "
            f"response_words={len(final_result.text.split())} "
            f"response_chars={len(final_result.text)} status={status} "
            f"drop_reason={','.join(final_drop_reasons) or '-'}",
            flush=True,
        )

    e2e_ms = seconds_to_ms(time.perf_counter() - request_start)
    status = "degraded" if bad_rounds else "success"
    return build_request_result(
        ok=True,
        status=status,
        request_id=request_id,
        audio_path=audio_path,
        language=language,
        audio=audio,
        sample_rate=sample_rate,
        e2e_ms=e2e_ms,
        first_round_ttft_ms=first_round_ttft_ms,
        round_ttft_ms=round_ttft_ms,
        request_latencies_ms=request_latencies_ms,
        rounds=rounds,
        total_uploaded_audio_ms=total_uploaded_audio_ms,
        total_uploaded_bytes=total_uploaded_bytes,
        final_text=state.stable_text,
        bad_rounds=bad_rounds,
        dropped_rounds=dropped_rounds,
        timeout_rounds=timeout_rounds,
        repeated_rounds=repeated_rounds,
        too_long_rounds=too_long_rounds,
        drop_reasons_all=drop_reasons_all,
        error=None,
    )


def build_request_result(
    *,
    ok: bool,
    status: str,
    request_id: int | None,
    audio_path: Path,
    language: str,
    audio: np.ndarray,
    sample_rate: int,
    e2e_ms: float,
    first_round_ttft_ms: float | None,
    round_ttft_ms: list[float],
    request_latencies_ms: list[float],
    rounds: int,
    total_uploaded_audio_ms: float,
    total_uploaded_bytes: int,
    final_text: str,
    bad_rounds: int,
    dropped_rounds: int,
    timeout_rounds: int,
    repeated_rounds: int,
    too_long_rounds: int,
    drop_reasons_all: list[str],
    error: str | None,
) -> dict[str, Any]:
    audio_duration_ms = len(audio) / sample_rate * 1000.0
    rtf = e2e_ms / audio_duration_ms if audio_duration_ms > 0 else None
    return {
        "ok": ok,
        "status": status,
        "request_id": request_id,
        "file": audio_path.name,
        "audio_path": str(audio_path.resolve()),
        "language": language,
        "audio_duration_ms": audio_duration_ms,
        "e2e_ms": e2e_ms,
        "ttft_ms": first_round_ttft_ms,
        "round_ttft_ms": round_ttft_ms,
        "request_latencies_ms": request_latencies_ms,
        "avg_round_latency_ms": mean(request_latencies_ms),
        "max_round_latency_ms": max(request_latencies_ms)
        if request_latencies_ms
        else None,
        "rounds": rounds,
        "http_requests": len(request_latencies_ms),
        "uploaded_audio_ms": total_uploaded_audio_ms,
        "uploaded_bytes": total_uploaded_bytes,
        "rtf": rtf,
        "final_text_words": len(final_text.split()),
        "final_text_chars": len(final_text),
        "bad_rounds": bad_rounds,
        "dropped_rounds": dropped_rounds,
        "timeout_rounds": timeout_rounds,
        "repeated_rounds": repeated_rounds,
        "too_long_rounds": too_long_rounds,
        "drop_reasons": dict(Counter(drop_reasons_all)),
        "error": error,
    }


def run_worker(
    request_id: int,
    audio_path: Path,
    args: argparse.Namespace,
    start_event: threading.Event,
) -> dict[str, Any]:
    start_event.wait()
    try:
        return run_one_request(
            request_id=request_id,
            audio_path=audio_path,
            args=args,
        )
    except Exception as exc:
        language = "unknown"
        try:
            language = infer_language(audio_path, args.language)
        except Exception:
            pass
        return {
            "ok": False,
            "status": "failed",
            "request_id": request_id,
            "file": audio_path.name,
            "audio_path": str(audio_path.resolve()),
            "language": language,
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
    for idx in range(args.warmup_iterations):
        try:
            result = run_one_request(
                request_id=-(idx + 1),
                audio_path=warmup_path,
                args=args,
                language_override=args.warmup_language,
            )
            times.append(result["e2e_ms"])
            print(
                f"  {result['status'].upper()} warmup {idx + 1}: "
                f"e2e_ms={result['e2e_ms']:.2f}, "
                f"ttft_ms={fmt_float(result['ttft_ms'])}, "
                f"http_requests={result['http_requests']}, "
                f"bad_rounds={result['bad_rounds']}"
            )
        except Exception as exc:
            print(f"  FAIL warmup {idx + 1}: {exc}")

    print("=" * 80)
    print("Warmup complete. Starting benchmark...\n")
    return times


def print_benchmark_result(
    *,
    results: list[dict[str, Any]],
    benchmark_duration_ms: float,
    concurrency: int,
) -> None:
    success_results = [r for r in results if r.get("status") == "success"]
    degraded_results = [r for r in results if r.get("status") == "degraded"]
    failed_results = [r for r in results if r.get("status") == "failed"]
    completed_results = success_results + degraded_results

    e2e_values = [r["e2e_ms"] for r in completed_results]
    ttft_values = [
        r["ttft_ms"] for r in completed_results if r.get("ttft_ms") is not None
    ]
    rtf_values = [r["rtf"] for r in completed_results if r.get("rtf") is not None]
    round_latency_values = [
        value
        for result in completed_results
        for value in result["request_latencies_ms"]
    ]
    uploaded_audio_ms = sum(r["uploaded_audio_ms"] for r in completed_results)
    source_audio_ms = sum(r["audio_duration_ms"] for r in completed_results)
    total_http_requests = sum(r["http_requests"] for r in completed_results)
    duration_s = benchmark_duration_ms / 1000.0

    print("=" * 60)
    print("          Arabic/Thai Prefix Streaming Benchmark")
    print("=" * 60)
    print_int_metric("Successful logical requests:", len(success_results))
    print_int_metric("Degraded logical requests:", len(degraded_results))
    print_int_metric("Failed logical requests:", len(failed_results))
    print_int_metric("Maximum logical concurrency:", concurrency)
    print_int_metric("Total HTTP audio requests:", total_http_requests)
    print_int_metric(
        "Dropped rounds:",
        sum(r.get("dropped_rounds", 0) for r in results),
    )
    print_int_metric(
        "Timeout rounds:",
        sum(r.get("timeout_rounds", 0) for r in results),
    )
    print_int_metric(
        "Repeated rounds:",
        sum(r.get("repeated_rounds", 0) for r in results),
    )
    print_int_metric(
        "Too-long rounds:",
        sum(r.get("too_long_rounds", 0) for r in results),
    )
    print_metric("Benchmark duration (ms):", benchmark_duration_ms, 2)
    print_metric(
        "Completed throughput (req/s):",
        safe_div(len(completed_results), duration_s),
        2,
    )
    print_metric(
        "HTTP request throughput (req/s):",
        safe_div(total_http_requests, duration_s),
        2,
    )
    print("-" * 60)
    print("                    End-to-End Latency")
    print("-" * 60)
    for key, value in summarize(e2e_values).items():
        print_metric(f"{key.upper()} E2E (ms):", value, 2)
    print("-" * 60)
    print("                 First Round Time to First Text")
    print("-" * 60)
    for key, value in summarize(ttft_values).items():
        print_metric(f"{key.upper()} TTFT (ms):", value, 2)
    print("-" * 60)
    print("                   Per HTTP Request Latency")
    print("-" * 60)
    for key, value in summarize(round_latency_values).items():
        print_metric(f"{key.upper()} round latency (ms):", value, 2)
    print("-" * 60)
    print("                            Audio")
    print("-" * 60)
    print_metric("Source audio total (ms):", source_audio_ms, 2)
    print_metric("Uploaded audio total (ms):", uploaded_audio_ms, 2)
    print_metric(
        "Upload amplification:",
        safe_div(uploaded_audio_ms, source_audio_ms),
        3,
    )
    print_metric(
        "Audio throughput (audio ms / wall ms):",
        safe_div(source_audio_ms, benchmark_duration_ms),
        3,
    )
    print_metric("Mean RTF:", mean(rtf_values), 3)
    print_metric("P99 RTF:", percentile(rtf_values, 99), 3)
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative prefix-streaming ASR benchmark for Arabic/Thai."
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
    )
    parser.add_argument(
        "--language",
        choices=("auto", "ar", "th"),
        default="auto",
        help="Use auto to infer ar/th from file names.",
    )
    parser.add_argument("--to-language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--max-audio-window-seconds", type=float, default=30.0)
    parser.add_argument("--max-prefix-words", type=int, default=100)
    parser.add_argument("--max-prefix-chars", type=int, default=500)
    parser.add_argument("--holdback-words", type=int, default=5)
    parser.add_argument("--holdback-chars", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--final-max-tokens", type=int, default=512)
    parser.add_argument("--max-response-words", type=int, default=500)
    parser.add_argument("--max-response-chars", type=int, default=3000)
    parser.add_argument("--max-request-ms", type=float, default=30000.0)
    parser.add_argument("--repeat-ngram-size", type=int, default=8)
    parser.add_argument("--repeat-ngram-threshold", type=int, default=5)
    parser.add_argument("--max-consecutive-bad-rounds", type=int, default=3)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log-rounds", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument(
        "--output-file",
        default="prefix_streaming_ar_th_benchmark.json",
    )
    parser.add_argument("--warmup-file", default=None)
    parser.add_argument(
        "--warmup-language",
        choices=("ar", "th"),
        default="ar",
        help="Language used for warmup when the warmup file name is generic.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")
    if args.max_consecutive_bad_rounds < 1:
        raise ValueError("--max-consecutive-bad-rounds must be >= 1")

    audio_paths = find_all_audio(args.audio_dir, args.language)
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

    print(f"Found {len(audio_paths)} Arabic/Thai benchmark files in {args.audio_dir}")
    print(f"Benchmark logical requests: {num_requests}")
    print(f"Maximum logical concurrency: {args.concurrency}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(
        "Conservative pseudo streaming: "
        f"chunk_ms={args.chunk_seconds * 1000:.0f}, "
        f"max_audio_window_ms={args.max_audio_window_seconds * 1000:.0f}, "
        f"max_prefix_words={args.max_prefix_words}, "
        f"max_prefix_chars={args.max_prefix_chars}, "
        f"holdback_words={args.holdback_words}, "
        f"holdback_chars={args.holdback_chars}, "
        f"max_tokens={args.max_tokens}, "
        f"max_request_ms={args.max_request_ms:.0f}, "
        f"stream={args.stream}"
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
                        "success": sum(1 for r in results if r["status"] == "success"),
                        "degraded": sum(
                            1 for r in results if r["status"] == "degraded"
                        ),
                        "failed": sum(1 for r in results if r["status"] == "failed"),
                        "file": result["file"],
                    }
                    if result.get("e2e_ms") is not None:
                        postfix["e2e_ms"] = f"{result['e2e_ms']:.2f}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)
            finally:
                pbar.close()
        else:
            for completed, future in enumerate(iterator, start=1):
                result = future.result()
                results.append(result)
                print(f"Progress: {completed}/{len(futures)} requests completed")

        benchmark_duration_ms = seconds_to_ms(time.perf_counter() - benchmark_start)

    print_benchmark_result(
        results=results,
        benchmark_duration_ms=benchmark_duration_ms,
        concurrency=args.concurrency,
    )

    completed_results = [r for r in results if r.get("status") != "failed"]
    output = {
        "config": vars(args),
        "warmup": {
            "performed": bool(warmup_times),
            "iterations": len(warmup_times),
            "e2e_ms": warmup_times,
            "avg_e2e_ms": mean(warmup_times),
        },
        "summary": {
            "successful_requests": sum(1 for r in results if r["status"] == "success"),
            "degraded_requests": sum(1 for r in results if r["status"] == "degraded"),
            "failed_requests": sum(1 for r in results if r["status"] == "failed"),
            "benchmark_duration_ms": benchmark_duration_ms,
            "e2e_ms": summarize([r["e2e_ms"] for r in completed_results]),
            "ttft_ms": summarize(
                [
                    r["ttft_ms"]
                    for r in completed_results
                    if r.get("ttft_ms") is not None
                ]
            ),
            "round_latency_ms": summarize(
                [
                    value
                    for result in completed_results
                    for value in result["request_latencies_ms"]
                ]
            ),
            "rtf": summarize(
                [r["rtf"] for r in completed_results if r.get("rtf") is not None]
            ),
            "dropped_rounds": sum(r.get("dropped_rounds", 0) for r in results),
            "timeout_rounds": sum(r.get("timeout_rounds", 0) for r in results),
            "repeated_rounds": sum(r.get("repeated_rounds", 0) for r in results),
            "too_long_rounds": sum(r.get("too_long_rounds", 0) for r in results),
        },
        "per_request": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
