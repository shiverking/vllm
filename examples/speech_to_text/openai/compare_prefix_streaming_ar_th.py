# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-audio ASR with batched prefix-streaming ASR for Arabic/Thai.

This script is for result-quality comparison, not throughput benchmarking. For
each Arabic or Thai audio file it runs:

1. A single full-audio request, used as the baseline.
2. A conservative batched prefix-streaming request.
3. Text similarity metrics with the single full-audio result as reference.

The batched path reuses the Arabic/Thai safety controls from
``bench_prefix_streaming_ar_th.py`` so repeated or overlong rounds do not poison
later ``response_prefix`` values.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm.multimodal.media.audio import load_audio

try:
    import bench_prefix_streaming_ar_th as safe_client
except ImportError:
    from examples.speech_to_text.openai import (
        bench_prefix_streaming_ar_th as safe_client,
    )

try:
    import openai_prefix_streaming_transcription_client as prefix_client
except ImportError:
    from examples.speech_to_text.openai import (
        openai_prefix_streaming_transcription_client as prefix_client,
    )


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for ref_idx, ref_char in enumerate(reference, start=1):
        current = [ref_idx]
        for hyp_idx, hyp_char in enumerate(hypothesis, start=1):
            substitution = previous[hyp_idx - 1] + int(ref_char != hyp_char)
            insertion = current[hyp_idx - 1] + 1
            deletion = previous[hyp_idx] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def sequence_ratio(reference: str, hypothesis: str) -> float:
    return difflib.SequenceMatcher(None, reference, hypothesis).ratio()


def error_rate(reference: str, hypothesis: str) -> float | None:
    if not reference:
        return None
    return levenshtein_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return None

    previous = list(range(len(hyp_words) + 1))
    for ref_idx, ref_word in enumerate(ref_words, start=1):
        current = [ref_idx]
        for hyp_idx, hyp_word in enumerate(hyp_words, start=1):
            substitution = previous[hyp_idx - 1] + int(ref_word != hyp_word)
            insertion = current[hyp_idx - 1] + 1
            deletion = previous[hyp_idx] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(ref_words)


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_similarity_metrics(
    baseline_text: str,
    batched_text: str,
) -> dict[str, Any]:
    baseline_norm = normalize_text(baseline_text)
    batched_norm = normalize_text(batched_text)
    baseline_compact = compact_text(baseline_text)
    batched_compact = compact_text(batched_text)

    return {
        "char_similarity": sequence_ratio(baseline_norm, batched_norm),
        "compact_char_similarity": sequence_ratio(
            baseline_compact,
            batched_compact,
        ),
        "word_similarity": sequence_ratio(
            " ".join(baseline_norm.split()),
            " ".join(batched_norm.split()),
        ),
        "cer": error_rate(baseline_compact, batched_compact),
        "wer": word_error_rate(baseline_norm, batched_norm),
        "baseline_chars": len(baseline_text),
        "batched_chars": len(batched_text),
        "baseline_words": len(baseline_text.split()),
        "batched_words": len(batched_text.split()),
    }


def post_request(
    *,
    args: argparse.Namespace,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    response_prefix: str,
    max_tokens: int,
    max_request_ms: float,
) -> safe_client.SafeRequestResult:
    return safe_client.post_safe_audio_request(
        api_base=args.api_base,
        endpoint=endpoint,
        audio=audio,
        sample_rate=sample_rate,
        model=args.model,
        language=language,
        to_language=args.to_language,
        response_prefix=response_prefix,
        stream=args.stream,
        temperature=args.temperature,
        timeout=args.timeout,
        max_completion_tokens=max_tokens,
        max_request_ms=max_request_ms,
        max_response_chars=args.max_response_chars,
        repeat_ngram_size=args.repeat_ngram_size,
        repeat_ngram_threshold=args.repeat_ngram_threshold,
    )


def run_single_full_audio(
    *,
    args: argparse.Namespace,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = post_request(
        args=args,
        endpoint=endpoint,
        audio=audio,
        sample_rate=sample_rate,
        language=language,
        response_prefix="",
        max_tokens=args.single_max_tokens,
        max_request_ms=args.single_max_request_ms,
    )
    e2e_ms = safe_client.seconds_to_ms(time.perf_counter() - start)
    drop_reasons = safe_client.build_drop_reasons(
        result,
        max_response_words=args.single_max_response_words,
        max_response_chars=args.max_response_chars,
    )
    return {
        "text": result.text,
        "e2e_ms": e2e_ms,
        "latency_ms": result.latency_ms,
        "ttft_ms": result.ttft_ms,
        "response_words": len(result.text.split()),
        "response_chars": len(result.text),
        "drop_reasons": drop_reasons,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
        "repeated": result.repeated,
    }


def run_batched_prefix_streaming(
    *,
    args: argparse.Namespace,
    endpoint: str,
    audio_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    chunk_samples = max(1, int(args.chunk_seconds * sample_rate))
    max_window_samples = (
        int(args.max_audio_window_seconds * sample_rate)
        if args.max_audio_window_seconds > 0
        else None
    )
    state = prefix_client.PrefixStreamingState()
    rounds: list[dict[str, Any]] = []
    consecutive_bad_rounds = 0
    aborted = False
    abort_reason: str | None = None

    for round_idx, end in enumerate(
        range(chunk_samples, len(audio) + chunk_samples, chunk_samples),
        start=1,
    ):
        end = min(end, len(audio))
        start = 0
        if max_window_samples is not None:
            start = max(0, end - max_window_samples)

        prefix = safe_client.cap_text_by_words_and_chars(
            state.stable_text,
            args.max_prefix_words,
            args.max_prefix_chars,
        )
        if args.log_rounds:
            print(
                "[batch-round-start] "
                f"file={audio_path.name} language={language} round={round_idx} "
                f"window_ms={start / sample_rate * 1000.0:.0f}-"
                f"{end / sample_rate * 1000.0:.0f} "
                f"prefix_chars={len(prefix)}",
                flush=True,
            )

        result = post_request(
            args=args,
            endpoint=endpoint,
            audio=audio[start:end],
            sample_rate=sample_rate,
            language=language,
            response_prefix=prefix,
            max_tokens=args.max_tokens,
            max_request_ms=args.max_request_ms,
        )
        drop_reasons = safe_client.build_drop_reasons(
            result,
            max_response_words=args.max_response_words,
            max_response_chars=args.max_response_chars,
        )
        if drop_reasons:
            consecutive_bad_rounds += 1
        else:
            consecutive_bad_rounds = 0
            safe_client.update_state_from_result(
                state,
                prefix,
                result.text,
                args.holdback_words,
                args.holdback_chars,
            )

        round_info = {
            "round": round_idx,
            "window_start_ms": start / sample_rate * 1000.0,
            "window_end_ms": end / sample_rate * 1000.0,
            "latency_ms": result.latency_ms,
            "ttft_ms": result.ttft_ms,
            "response_words": len(result.text.split()),
            "response_chars": len(result.text),
            "prefix_words": len(prefix.split()),
            "prefix_chars": len(prefix),
            "drop_reasons": drop_reasons,
        }
        rounds.append(round_info)

        if args.log_rounds:
            print(
                "[batch-round-done] "
                f"file={audio_path.name} round={round_idx} "
                f"latency_ms={result.latency_ms:.0f} "
                f"response_chars={len(result.text)} "
                f"drop_reason={','.join(drop_reasons) or '-'}",
                flush=True,
            )

        if consecutive_bad_rounds >= args.max_consecutive_bad_rounds:
            aborted = True
            abort_reason = "max consecutive bad rounds reached"
            break
        if end >= len(audio):
            break

    if not aborted:
        final_prefix = safe_client.cap_text_by_words_and_chars(
            state.stable_text,
            args.max_prefix_words,
            args.max_prefix_chars,
        )
        final_audio = audio
        if max_window_samples is not None:
            final_audio = audio[-max_window_samples:]

        final_result = post_request(
            args=args,
            endpoint=endpoint,
            audio=final_audio,
            sample_rate=sample_rate,
            language=language,
            response_prefix=final_prefix,
            max_tokens=args.final_max_tokens,
            max_request_ms=args.max_request_ms,
        )
        final_drop_reasons = safe_client.build_drop_reasons(
            final_result,
            max_response_words=args.max_response_words,
            max_response_chars=args.max_response_chars,
        )
        if final_drop_reasons:
            rounds.append(
                {
                    "round": "final",
                    "latency_ms": final_result.latency_ms,
                    "ttft_ms": final_result.ttft_ms,
                    "response_words": len(final_result.text.split()),
                    "response_chars": len(final_result.text),
                    "prefix_words": len(final_prefix.split()),
                    "prefix_chars": len(final_prefix),
                    "drop_reasons": final_drop_reasons,
                }
            )
        else:
            state.stable_text = prefix_client.merge_history_and_candidate(
                state.stable_text,
                prefix_client.merge_prefix_and_response(
                    final_prefix,
                    final_result.text,
                ),
            )
            state.unstable_text = ""
            rounds.append(
                {
                    "round": "final",
                    "latency_ms": final_result.latency_ms,
                    "ttft_ms": final_result.ttft_ms,
                    "response_words": len(final_result.text.split()),
                    "response_chars": len(final_result.text),
                    "prefix_words": len(final_prefix.split()),
                    "prefix_chars": len(final_prefix),
                    "drop_reasons": [],
                }
            )

    dropped_rounds = [r for r in rounds if r["drop_reasons"]]
    return {
        "text": state.stable_text,
        "e2e_ms": safe_client.seconds_to_ms(time.perf_counter() - start_time),
        "status": "failed" if aborted else "degraded" if dropped_rounds else "success",
        "aborted": aborted,
        "abort_reason": abort_reason,
        "rounds": rounds,
        "round_count": len([r for r in rounds if r["round"] != "final"]),
        "dropped_rounds": len(dropped_rounds),
        "drop_reasons": dict(
            safe_client.Counter(
                reason for r in dropped_rounds for reason in r["drop_reasons"]
            )
        ),
    }


def compare_one_audio(
    audio_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    language = safe_client.infer_language(audio_path, args.language)
    endpoint = "translations" if args.task == "translate" else "transcriptions"
    audio, sample_rate = load_audio(str(audio_path), sr=args.sample_rate, mono=True)
    duration_ms = len(audio) / sample_rate * 1000.0

    print(
        "\n"
        f"[compare] file={audio_path.name} language={language} "
        f"duration_ms={duration_ms:.0f}",
        flush=True,
    )

    single = run_single_full_audio(
        args=args,
        endpoint=endpoint,
        audio=audio,
        sample_rate=sample_rate,
        language=language,
    )
    print(
        "[single] "
        f"latency_ms={single['latency_ms']:.0f} "
        f"chars={single['response_chars']} "
        f"words={single['response_words']} "
        f"drop_reason={','.join(single['drop_reasons']) or '-'}",
        flush=True,
    )

    batched = run_batched_prefix_streaming(
        args=args,
        endpoint=endpoint,
        audio_path=audio_path,
        audio=audio,
        sample_rate=sample_rate,
        language=language,
    )
    metrics = build_similarity_metrics(single["text"], batched["text"])
    print(
        "[compare-result] "
        f"status={batched['status']} "
        f"char_similarity={metrics['char_similarity']:.4f} "
        f"compact_char_similarity={metrics['compact_char_similarity']:.4f} "
        f"cer={safe_client.fmt_float(metrics['cer'], 4)} "
        f"dropped_rounds={batched['dropped_rounds']}",
        flush=True,
    )

    result: dict[str, Any] = {
        "file": audio_path.name,
        "audio_path": str(audio_path.resolve()),
        "language": language,
        "audio_duration_ms": duration_ms,
        "single": {k: v for k, v in single.items() if k != "text"},
        "batched": {k: v for k, v in batched.items() if k != "text"},
        "similarity": metrics,
    }
    if args.include_text:
        result["single"]["text"] = single["text"]
        result["batched"]["text"] = batched["text"]
    return result


def print_summary(results: list[dict[str, Any]]) -> None:
    similarities = [
        r["similarity"]["compact_char_similarity"]
        for r in results
        if r["similarity"]["compact_char_similarity"] is not None
    ]
    cers = [
        r["similarity"]["cer"]
        for r in results
        if r["similarity"]["cer"] is not None
    ]
    degraded = sum(1 for r in results if r["batched"]["status"] == "degraded")
    failed = sum(1 for r in results if r["batched"]["status"] == "failed")
    dropped_rounds = sum(r["batched"]["dropped_rounds"] for r in results)

    print("\n" + "=" * 64)
    print(" Arabic/Thai Full-vs-Batched Similarity Summary")
    print("=" * 64)
    print(f"Files compared: {len(results)}")
    print(f"Batched degraded: {degraded}")
    print(f"Batched failed: {failed}")
    print(f"Dropped batched rounds: {dropped_rounds}")
    print(f"Compact char similarity: {summarize(similarities)}")
    print(f"CER: {summarize(cers)}")
    print("=" * 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-audio and batched prefix-streaming ASR."
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
    )
    parser.add_argument("--to-language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--max-audio-window-seconds", type=float, default=30.0)
    parser.add_argument("--max-prefix-words", type=int, default=100)
    parser.add_argument("--max-prefix-chars", type=int, default=500)
    parser.add_argument("--holdback-words", type=int, default=5)
    parser.add_argument("--holdback-chars", type=int, default=80)
    parser.add_argument("--single-max-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--final-max-tokens", type=int, default=512)
    parser.add_argument("--single-max-response-words", type=int, default=5000)
    parser.add_argument("--max-response-words", type=int, default=500)
    parser.add_argument("--max-response-chars", type=int, default=12000)
    parser.add_argument("--single-max-request-ms", type=float, default=120000.0)
    parser.add_argument("--max-request-ms", type=float, default=30000.0)
    parser.add_argument("--repeat-ngram-size", type=int, default=8)
    parser.add_argument("--repeat-ngram-threshold", type=int, default=5)
    parser.add_argument("--max-consecutive-bad-rounds", type=int, default=3)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--log-rounds", action="store_true")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full single and batched texts in the JSON output.",
    )
    parser.add_argument(
        "--output-file",
        default="prefix_streaming_ar_th_similarity.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_paths = safe_client.find_all_audio(args.audio_dir, args.language)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        audio_paths = audio_paths[: args.limit]

    print(f"Found {len(audio_paths)} Arabic/Thai files in {args.audio_dir}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(
        "Compare config: "
        f"chunk_ms={args.chunk_seconds * 1000:.0f}, "
        f"max_audio_window_ms={args.max_audio_window_seconds * 1000:.0f}, "
        f"single_max_tokens={args.single_max_tokens}, "
        f"batch_max_tokens={args.max_tokens}, "
        f"stream={args.stream}"
    )

    results: list[dict[str, Any]] = []
    for audio_path in audio_paths:
        results.append(compare_one_audio(audio_path.resolve(), args))

    print_summary(results)
    output = {
        "config": vars(args),
        "summary": {
            "files": len(results),
            "compact_char_similarity": summarize(
                [r["similarity"]["compact_char_similarity"] for r in results]
            ),
            "cer": summarize(
                [
                    r["similarity"]["cer"]
                    for r in results
                    if r["similarity"]["cer"] is not None
                ]
            ),
            "degraded": sum(
                1 for r in results if r["batched"]["status"] == "degraded"
            ),
            "failed": sum(1 for r in results if r["batched"]["status"] == "failed"),
            "dropped_rounds": sum(r["batched"]["dropped_rounds"] for r in results),
        },
        "per_file": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
