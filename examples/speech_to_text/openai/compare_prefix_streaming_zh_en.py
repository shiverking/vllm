# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare full-audio ASR with batched prefix-streaming ASR for Chinese/English.

This script is for result-quality comparison, not throughput benchmarking. For
each Chinese or English audio file it runs:

1. A single full-audio request, used as the baseline.
2. A batched prefix-streaming request.
3. Similarity metrics with the single full-audio result as reference.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import unicodedata
from collections import Counter
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


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".webm"}
SUPPORTED_LANGUAGES = {"zh", "en"}


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def content_text(text: str) -> str:
    chars: list[str] = []
    for char in compact_text(text):
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            continue
        chars.append(char)
    return "".join(chars)


def infer_optional_language(audio_path: Path, configured_language: str) -> str | None:
    try:
        return infer_language(audio_path, configured_language)
    except ValueError:
        return None


def infer_language(audio_path: Path, configured_language: str) -> str:
    if configured_language in SUPPORTED_LANGUAGES:
        return configured_language

    name = audio_path.stem.lower()
    if name.startswith("zh") or name.startswith("cn") or "chinese" in name:
        return "zh"
    if name.startswith("en") or "english" in name:
        return "en"
    raise ValueError(
        f"cannot infer Chinese/English language from file name: {audio_path.name}. "
        "Use --language zh or --language en."
    )


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
        raise RuntimeError(f"no Chinese/English audio files found in {audio_dir}")
    return files


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
    stable_len = min(len(stable_by_words), len(stable_by_chars))
    return text[:stable_len].rstrip(), text[stable_len:]


def split_with_language_holdback(
    text: str,
    language: str,
    holdback_words: int,
    holdback_chars: int,
) -> tuple[str, str]:
    if language == "zh":
        if holdback_chars <= 0:
            return text, ""
        if len(text) <= holdback_chars:
            return "", text
        return text[:-holdback_chars].rstrip(), text[-holdback_chars:]

    return split_with_dual_holdback(text, holdback_words, 0)


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


def fmt_float(value: float | None, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def build_similarity_metrics(
    baseline_text: str,
    batched_text: str,
) -> dict[str, Any]:
    baseline_norm = normalize_text(baseline_text)
    batched_norm = normalize_text(batched_text)
    baseline_compact = compact_text(baseline_text)
    batched_compact = compact_text(batched_text)
    baseline_content = content_text(baseline_text)
    batched_content = content_text(batched_text)

    return {
        "char_similarity": sequence_ratio(baseline_norm, batched_norm),
        "compact_char_similarity": sequence_ratio(
            baseline_compact,
            batched_compact,
        ),
        "content_similarity": sequence_ratio(
            baseline_content,
            batched_content,
        ),
        "word_similarity": sequence_ratio(
            " ".join(baseline_norm.split()),
            " ".join(batched_norm.split()),
        ),
        "cer": error_rate(baseline_compact, batched_compact),
        "content_cer": error_rate(baseline_content, batched_content),
        "wer": word_error_rate(baseline_norm, batched_norm),
        "baseline_chars": len(baseline_text),
        "batched_chars": len(batched_text),
        "baseline_words": len(baseline_text.split()),
        "batched_words": len(batched_text.split()),
    }


def extract_stream_text(response: requests.Response) -> tuple[str, float | None]:
    text_parts: list[str] = []
    ttft_ms: float | None = None
    request_start = time.perf_counter()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            line = line[len("data: ") :]
        if line.strip() == "[DONE]":
            break
        payload = json.loads(line)
        delta = prefix_client.extract_stream_delta(payload)
        if delta:
            if ttft_ms is None:
                ttft_ms = seconds_to_ms(time.perf_counter() - request_start)
            text_parts.append(delta)
    return "".join(text_parts), ttft_ms


def post_audio_request(
    *,
    args: argparse.Namespace,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
    response_prefix: str,
    max_tokens: int,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model": args.model,
        "response_format": "json",
        "response_prefix": response_prefix,
        "stream": "true" if args.stream else "false",
        "temperature": args.temperature,
        "max_completion_tokens": max_tokens,
        "language": language,
    }
    if args.to_language:
        data["to_language"] = args.to_language

    request_start = time.perf_counter()
    with prefix_client.audio_to_wav_buffer(audio, sample_rate) as wav_buffer:
        upload_bytes = wav_buffer.getbuffer().nbytes
        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        with requests.post(
            prefix_client.build_audio_endpoint_url(args.api_base, endpoint),
            data=data,
            files=files,
            stream=args.stream,
            timeout=args.timeout,
        ) as response:
            response.raise_for_status()
            if args.stream:
                text, ttft_ms = extract_stream_text(response)
            else:
                text = response.json()["text"]
                ttft_ms = None

    latency_ms = seconds_to_ms(time.perf_counter() - request_start)
    repeated = has_repeated_char_ngram(
        text,
        args.repeat_ngram_size,
        args.repeat_ngram_threshold,
    )
    return {
        "text": text,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "upload_audio_ms": seconds_to_ms(len(audio) / sample_rate),
        "upload_bytes": upload_bytes,
        "repeated": repeated,
    }


def build_drop_reasons(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if args.drop_repeated and result["repeated"]:
        reasons.append("repeated")
    if args.max_response_words > 0:
        if len(result["text"].split()) > args.max_response_words:
            reasons.append("too_many_words")
    if args.max_response_chars > 0 and len(result["text"]) > args.max_response_chars:
        reasons.append("too_many_chars")
    return reasons


def run_single_full_audio(
    *,
    args: argparse.Namespace,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    language: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = post_audio_request(
        args=args,
        endpoint=endpoint,
        audio=audio,
        sample_rate=sample_rate,
        language=language,
        response_prefix="",
        max_tokens=args.single_max_tokens,
    )
    result["e2e_ms"] = seconds_to_ms(time.perf_counter() - start)
    result["response_words"] = len(result["text"].split())
    result["response_chars"] = len(result["text"])
    return result


def update_state_from_result(
    state: prefix_client.PrefixStreamingState,
    prefix: str,
    result_text: str,
    language: str,
    args: argparse.Namespace,
) -> None:
    candidate_text = prefix_client.merge_history_and_candidate(
        state.stable_text,
        prefix_client.merge_prefix_and_response(prefix, result_text),
    )
    state.stable_text, state.unstable_text = split_with_language_holdback(
        candidate_text,
        language,
        args.holdback_words,
        args.holdback_chars,
    )


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

    for round_idx, end in enumerate(
        range(chunk_samples, len(audio) + chunk_samples, chunk_samples),
        start=1,
    ):
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
                "[batch-round-start] "
                f"file={audio_path.name} language={language} round={round_idx} "
                f"window_ms={start / sample_rate * 1000.0:.0f}-"
                f"{end / sample_rate * 1000.0:.0f} "
                f"prefix_words={len(prefix.split())} "
                f"prefix_chars={len(prefix)}",
                flush=True,
            )

        result = post_audio_request(
            args=args,
            endpoint=endpoint,
            audio=audio[start:end],
            sample_rate=sample_rate,
            language=language,
            response_prefix=prefix,
            max_tokens=args.max_tokens,
        )
        drop_reasons = build_drop_reasons(result, args)
        if not drop_reasons:
            update_state_from_result(
                state,
                prefix,
                result["text"],
                language,
                args,
            )

        round_info = {
            "round": round_idx,
            "window_start_ms": start / sample_rate * 1000.0,
            "window_end_ms": end / sample_rate * 1000.0,
            "latency_ms": result["latency_ms"],
            "ttft_ms": result["ttft_ms"],
            "response_words": len(result["text"].split()),
            "response_chars": len(result["text"]),
            "prefix_words": len(prefix.split()),
            "prefix_chars": len(prefix),
            "repeated": result["repeated"],
            "drop_reasons": drop_reasons,
        }
        rounds.append(round_info)

        if args.log_rounds:
            print(
                "[batch-round-done] "
                f"file={audio_path.name} round={round_idx} "
                f"latency_ms={result['latency_ms']:.0f} "
                f"response_chars={len(result['text'])} "
                f"drop_reason={','.join(drop_reasons) or '-'}",
                flush=True,
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

    final_result = post_audio_request(
        args=args,
        endpoint=endpoint,
        audio=final_audio,
        sample_rate=sample_rate,
        language=language,
        response_prefix=final_prefix,
        max_tokens=args.final_max_tokens,
    )
    final_drop_reasons = build_drop_reasons(final_result, args)
    if not final_drop_reasons:
        state.stable_text = prefix_client.merge_history_and_candidate(
            state.stable_text,
            prefix_client.merge_prefix_and_response(
                final_prefix,
                final_result["text"],
            ),
        )
        state.unstable_text = ""

    rounds.append(
        {
            "round": "final",
            "latency_ms": final_result["latency_ms"],
            "ttft_ms": final_result["ttft_ms"],
            "response_words": len(final_result["text"].split()),
            "response_chars": len(final_result["text"]),
            "prefix_words": len(final_prefix.split()),
            "prefix_chars": len(final_prefix),
            "repeated": final_result["repeated"],
            "drop_reasons": final_drop_reasons,
        }
    )

    dropped_rounds = [r for r in rounds if r["drop_reasons"]]
    return {
        "text": state.display_text,
        "e2e_ms": seconds_to_ms(time.perf_counter() - start_time),
        "status": "degraded" if dropped_rounds else "success",
        "rounds": rounds,
        "round_count": len([r for r in rounds if r["round"] != "final"]),
        "dropped_rounds": len(dropped_rounds),
        "drop_reasons": dict(
            Counter(reason for r in dropped_rounds for reason in r["drop_reasons"])
        ),
    }


def compare_one_audio(
    audio_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    language = infer_language(audio_path, args.language)
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
        f"repeated={single['repeated']}",
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
        f"content_similarity={metrics['content_similarity']:.4f} "
        f"cer={fmt_float(metrics['cer'])} "
        f"content_cer={fmt_float(metrics['content_cer'])} "
        f"wer={fmt_float(metrics['wer'])} "
        f"dropped_rounds={batched['dropped_rounds']}",
        flush=True,
    )
    print("\n[baseline-text]")
    print(single["text"])
    print("\n[batched-final-text]")
    print(batched["text"])

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
    compact_similarities = [
        r["similarity"]["compact_char_similarity"]
        for r in results
        if r["similarity"]["compact_char_similarity"] is not None
    ]
    content_similarities = [
        r["similarity"]["content_similarity"]
        for r in results
        if r["similarity"]["content_similarity"] is not None
    ]
    cers = [
        r["similarity"]["cer"]
        for r in results
        if r["similarity"]["cer"] is not None
    ]
    wers = [
        r["similarity"]["wer"]
        for r in results
        if r["similarity"]["wer"] is not None
    ]
    content_cers = [
        r["similarity"]["content_cer"]
        for r in results
        if r["similarity"]["content_cer"] is not None
    ]
    degraded = sum(1 for r in results if r["batched"]["status"] == "degraded")
    dropped_rounds = sum(r["batched"]["dropped_rounds"] for r in results)

    print("\n" + "=" * 64)
    print(" Chinese/English Full-vs-Batched Similarity Summary")
    print("=" * 64)
    print(f"Files compared: {len(results)}")
    print(f"Batched degraded: {degraded}")
    print(f"Dropped batched rounds: {dropped_rounds}")
    print(f"Compact char similarity: {summarize(compact_similarities)}")
    print(f"Content similarity: {summarize(content_similarities)}")
    print(f"CER: {summarize(cers)}")
    print(f"Content CER: {summarize(content_cers)}")
    print(f"WER: {summarize(wers)}")
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
        choices=("auto", "zh", "en"),
        default="auto",
    )
    parser.add_argument("--to-language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--max-audio-window-seconds", type=float, default=30.0)
    parser.add_argument("--max-prefix-words", type=int, default=100)
    parser.add_argument("--max-prefix-chars", type=int, default=800)
    parser.add_argument("--holdback-words", type=int, default=5)
    parser.add_argument(
        "--holdback-chars",
        type=int,
        default=80,
        help="Used for Chinese holdback. English keeps word holdback by default.",
    )
    parser.add_argument("--single-max-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--final-max-tokens", type=int, default=1024)
    parser.add_argument("--max-response-words", type=int, default=1000)
    parser.add_argument("--max-response-chars", type=int, default=12000)
    parser.add_argument("--repeat-ngram-size", type=int, default=12)
    parser.add_argument("--repeat-ngram-threshold", type=int, default=8)
    parser.add_argument(
        "--drop-repeated",
        action="store_true",
        help="Drop repeated rounds. Defaults off for Chinese/English comparison.",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-rounds", action="store_true")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full single and batched texts in the JSON output.",
    )
    parser.add_argument(
        "--output-file",
        default="prefix_streaming_zh_en_similarity.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_paths = find_all_audio(args.audio_dir, args.language)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        audio_paths = audio_paths[: args.limit]

    print(f"Found {len(audio_paths)} Chinese/English files in {args.audio_dir}")
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
            "content_similarity": summarize(
                [r["similarity"]["content_similarity"] for r in results]
            ),
            "content_cer": summarize(
                [
                    r["similarity"]["content_cer"]
                    for r in results
                    if r["similarity"]["content_cer"] is not None
                ]
            ),
            "wer": summarize(
                [
                    r["similarity"]["wer"]
                    for r in results
                    if r["similarity"]["wer"] is not None
                ]
            ),
            "degraded": sum(
                1 for r in results if r["batched"]["status"] == "degraded"
            ),
            "dropped_rounds": sum(r["batched"]["dropped_rounds"] for r in results),
        },
        "per_file": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
