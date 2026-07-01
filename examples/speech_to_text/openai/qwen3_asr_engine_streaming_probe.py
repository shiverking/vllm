# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe Qwen3-ASR streaming-cache feasibility.

This is an experimental diagnostic tool for Engine-level ASR streaming work.
It does not implement true cache reuse. Instead, it answers two Phase 1/2
questions before touching the vLLM scheduler:

1. Does appending audio insert new audio tokens before the assistant text?
2. How do full-audio and prefix-feedback pseudo-streaming outputs differ?

If new audio tokens are inserted before the assistant generation region, a
naive reuse of decoder KV for already generated transcript tokens is unsafe:
the cached tokens were produced with a shorter audio prefix and different
positions/attention context.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from vllm.multimodal.media.audio import load_audio


@dataclass
class ProbeTranscription:
    name: str
    text: str
    latency_ms: float
    ttft_ms: float | None
    audio_start_ms: float
    audio_end_ms: float
    response_words: int
    response_chars: int
    repeated: bool


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def audio_to_wav_buffer(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    buffer.seek(0)
    buffer.name = "audio.wav"
    return buffer


def build_audio_endpoint_url(api_base: str, endpoint: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/audio"):
        return f"{base}/{endpoint}"
    return f"{base}/audio/{endpoint}"


def extract_stream_delta(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content is None:
        return ""
    return content


def normalize_text(text: str) -> str:
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def content_text(text: str) -> str:
    chars: list[str] = []
    compact = re.sub(r"\s+", "", normalize_text(text))
    for char in compact:
        if unicodedata.category(char)[0] in {"P", "S"}:
            continue
        chars.append(char)
    return "".join(chars)


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


def content_cer(reference: str, hypothesis: str) -> float | None:
    reference_content = content_text(reference)
    hypothesis_content = content_text(hypothesis)
    if not reference_content:
        return None
    return (
        levenshtein_distance(reference_content, hypothesis_content)
        / len(reference_content)
    )


def content_similarity(reference: str, hypothesis: str) -> float:
    import difflib

    return difflib.SequenceMatcher(
        None,
        content_text(reference),
        content_text(hypothesis),
    ).ratio()


def has_repeated_char_ngram(
    text: str,
    ngram_size: int,
    threshold: int,
) -> bool:
    if ngram_size <= 0 or threshold <= 0:
        return False

    compact = re.sub(r"\s+", "", text)
    if len(compact) < ngram_size:
        return False

    counts: dict[str, int] = {}
    for idx in range(0, len(compact) - ngram_size + 1):
        ngram = compact[idx : idx + ngram_size]
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] > threshold:
            return True
    return False


def approximate_audio_frames(num_samples: int, sample_rate: int) -> int:
    # Whisper-style feature extractors use roughly 100 frames per second.
    return int(np.ceil(num_samples / sample_rate * 100.0))


def qwen3_asr_audio_tokens_from_frames(num_frames: int) -> int:
    leave = num_frames % 100
    feat_lengths = (leave - 1) // 2 + 1 if leave > 0 else 0
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
        if feat_lengths > 0
        else 0
    )
    return int(output_lengths + (num_frames // 100) * 13)


def estimate_audio_token_count(num_samples: int, sample_rate: int) -> dict[str, int]:
    frames = approximate_audio_frames(num_samples, sample_rate)
    return {
        "audio_samples": num_samples,
        "approx_feature_frames": frames,
        "approx_qwen3_asr_audio_tokens": qwen3_asr_audio_tokens_from_frames(frames),
    }


def analyze_decoder_kv_reuse(
    prefix_samples: int,
    target_samples: int,
    sample_rate: int,
) -> dict[str, Any]:
    prefix = estimate_audio_token_count(prefix_samples, sample_rate)
    target = estimate_audio_token_count(target_samples, sample_rate)
    inserted_audio_tokens = (
        target["approx_qwen3_asr_audio_tokens"]
        - prefix["approx_qwen3_asr_audio_tokens"]
    )
    return {
        "prefix_audio": prefix,
        "target_audio": target,
        "inserted_audio_tokens_before_assistant": inserted_audio_tokens,
        "naive_decoder_kv_reuse_safe": inserted_audio_tokens == 0,
        "verdict": (
            "unsafe: appending audio inserts new audio tokens before assistant "
            "generation, so cached transcript-token KV from the shorter audio "
            "was computed with different positions/attention context"
            if inserted_audio_tokens > 0
            else "possibly safe: no new audio tokens are inserted before assistant"
        ),
    }


def extract_stream_text(response: requests.Response) -> tuple[str, float | None]:
    parts: list[str] = []
    ttft_ms: float | None = None
    start = time.perf_counter()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            line = line[len("data: ") :]
        if line.strip() == "[DONE]":
            break
        payload = json.loads(line)
        delta = extract_stream_delta(payload)
        if delta:
            if ttft_ms is None:
                ttft_ms = seconds_to_ms(time.perf_counter() - start)
            parts.append(delta)
    return "".join(parts), ttft_ms


def post_audio_request(
    *,
    args: argparse.Namespace,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    response_prefix: str,
) -> tuple[str, float, float | None, int]:
    data: dict[str, Any] = {
        "model": args.model,
        "response_format": "json",
        "response_prefix": response_prefix,
        "stream": "true" if args.stream else "false",
        "temperature": args.temperature,
    }
    if args.language:
        data["language"] = args.language
    if args.max_completion_tokens is not None:
        data["max_completion_tokens"] = args.max_completion_tokens

    request_start = time.perf_counter()
    with audio_to_wav_buffer(audio, sample_rate) as wav_buffer:
        upload_bytes = wav_buffer.getbuffer().nbytes
        files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
        with requests.post(
            build_audio_endpoint_url(args.api_base, endpoint),
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
    return text, latency_ms, ttft_ms, upload_bytes


def transcribe_slice(
    *,
    name: str,
    args: argparse.Namespace,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    response_prefix: str = "",
) -> ProbeTranscription:
    text, latency_ms, ttft_ms, _ = post_audio_request(
        args=args,
        endpoint=endpoint,
        audio=audio[start_sample:end_sample],
        sample_rate=sample_rate,
        response_prefix=response_prefix,
    )
    return ProbeTranscription(
        name=name,
        text=text,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        audio_start_ms=start_sample / sample_rate * 1000.0,
        audio_end_ms=end_sample / sample_rate * 1000.0,
        response_words=len(text.split()),
        response_chars=len(text),
        repeated=has_repeated_char_ngram(
            text,
            args.repeat_ngram_size,
            args.repeat_ngram_threshold,
        ),
    )


def compare_outputs(
    reference: ProbeTranscription,
    hypothesis: ProbeTranscription,
) -> dict[str, Any]:
    return {
        "reference": reference.name,
        "hypothesis": hypothesis.name,
        "content_similarity": content_similarity(reference.text, hypothesis.text),
        "content_cer": content_cer(reference.text, hypothesis.text),
        "reference_chars": len(reference.text),
        "hypothesis_chars": len(hypothesis.text),
        "reference_repeated": reference.repeated,
        "hypothesis_repeated": hypothesis.repeated,
    }


def print_transcription(result: ProbeTranscription, include_text: bool) -> None:
    print(
        f"[{result.name}] "
        f"audio_ms={result.audio_start_ms:.0f}-{result.audio_end_ms:.0f} "
        f"latency_ms={result.latency_ms:.0f} "
        f"ttft_ms={result.ttft_ms if result.ttft_ms is not None else -1:.0f} "
        f"chars={result.response_chars} words={result.response_words} "
        f"repeated={result.repeated}",
        flush=True,
    )
    if include_text:
        print(f"[{result.name}-text]")
        print(result.text)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio_path)
    audio, sample_rate = load_audio(str(audio_path), sr=args.sample_rate, mono=True)
    duration_s = len(audio) / sample_rate
    prefix_end = min(len(audio), int(args.prefix_seconds * sample_rate))
    target_end = min(len(audio), int(args.target_seconds * sample_rate))
    if target_end <= prefix_end:
        raise ValueError("--target-seconds must be greater than --prefix-seconds")

    print(
        "[setup] "
        f"audio={audio_path} duration_ms={duration_s * 1000.0:.0f} "
        f"sample_rate={sample_rate} prefix_ms={args.prefix_seconds * 1000.0:.0f} "
        f"target_ms={args.target_seconds * 1000.0:.0f}",
        flush=True,
    )

    kv_analysis = analyze_decoder_kv_reuse(prefix_end, target_end, sample_rate)
    print("[cache-analysis]")
    print(json.dumps(kv_analysis, indent=2, ensure_ascii=False))

    endpoint = "translations" if args.task == "translate" else "transcriptions"
    results: list[ProbeTranscription] = []
    if not args.no_server_probe:
        prefix_result = transcribe_slice(
            name="prefix_full_decode",
            args=args,
            endpoint=endpoint,
            audio=audio,
            sample_rate=sample_rate,
            start_sample=0,
            end_sample=prefix_end,
        )
        target_result = transcribe_slice(
            name="target_full_decode",
            args=args,
            endpoint=endpoint,
            audio=audio,
            sample_rate=sample_rate,
            start_sample=0,
            end_sample=target_end,
        )
        appended_only_result = transcribe_slice(
            name="appended_audio_only_decode",
            args=args,
            endpoint=endpoint,
            audio=audio,
            sample_rate=sample_rate,
            start_sample=prefix_end,
            end_sample=target_end,
        )
        results.extend([prefix_result, target_result, appended_only_result])

        if args.probe_response_prefix:
            prefixed_result = transcribe_slice(
                name="target_with_text_prefix_decode",
                args=args,
                endpoint=endpoint,
                audio=audio,
                sample_rate=sample_rate,
                start_sample=0,
                end_sample=target_end,
                response_prefix=prefix_result.text,
            )
            results.append(prefixed_result)

        for result in results:
            print_transcription(result, args.include_text)

    comparisons = []
    if len(results) >= 3:
        target = results[1]
        comparisons.append(compare_outputs(target, results[0]))
        comparisons.append(compare_outputs(target, results[2]))
        if len(results) > 3:
            comparisons.append(compare_outputs(target, results[3]))
        print("[comparisons]")
        print(json.dumps(comparisons, indent=2, ensure_ascii=False))

    output = {
        "config": vars(args),
        "audio": {
            "path": str(audio_path.resolve()),
            "duration_ms": duration_s * 1000.0,
            "sample_rate": sample_rate,
            "prefix_end_ms": prefix_end / sample_rate * 1000.0,
            "target_end_ms": target_end / sample_rate * 1000.0,
        },
        "cache_analysis": kv_analysis,
        "transcriptions": [
            {
                **result.__dict__,
                "text": result.text if args.include_text else None,
            }
            for result in results
        ],
        "comparisons": comparisons,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3-ASR Engine-level streaming feasibility."
    )
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--prefix-seconds", type=float, default=10.0)
    parser.add_argument("--target-seconds", type=float, default=20.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--repeat-ngram-size", type=int, default=12)
    parser.add_argument("--repeat-ngram-threshold", type=int, default=8)
    parser.add_argument(
        "--probe-response-prefix",
        action="store_true",
        help=(
            "Also decode target audio with prefix_full_decode text as "
            "response_prefix."
        ),
    )
    parser.add_argument(
        "--no-server-probe",
        action="store_true",
        help="Only print token-layout cache analysis; do not call the server.",
    )
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument(
        "--output-file",
        default="qwen3_asr_engine_streaming_probe.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_probe(args)
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
