# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe Qwen3-ASR streaming semantics with an external session wrapper.

This script does not depend on vLLM's resumable request path. It uses normal
local AsyncLLM requests, while the wrapper keeps audio history, transcript
state, short text tails, and overlap-based delta merging outside the Engine.

The goal is to validate whether Qwen3-ASR can produce stable incremental ASR
semantics before investing in lower-level KV/cache plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import string
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    _ASR_TEXT_TAG,
)
from vllm.multimodal.media.audio import load_audio
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM


_PROTOCOL_PREFIX_RE = re.compile(
    rf"(?:^|\s*)language\s+[^<\r\n]{{1,80}}{re.escape(_ASR_TEXT_TAG)}",
    flags=re.IGNORECASE,
)
_INCOMPLETE_PROTOCOL_PREFIX_RE = re.compile(
    r"(?:^|\s*)language(?:\s+[^<\r\n]{0,80})?$",
    flags=re.IGNORECASE,
)


def skip_general_plugins_for_probe() -> None:
    import vllm.engine.arg_utils as arg_utils
    import vllm.plugins as plugins

    def load_no_general_plugins() -> None:
        plugins.plugins_loaded = True

    plugins.load_general_plugins = load_no_general_plugins
    arg_utils.load_general_plugins = load_no_general_plugins


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def strip_qwen3_asr_protocol_text(text: str) -> str:
    if not text:
        return ""

    text = _PROTOCOL_PREFIX_RE.sub("", text)
    text = _INCOMPLETE_PROTOCOL_PREFIX_RE.sub("", text)
    return text.replace(_ASR_TEXT_TAG, "")


def sanitize_context(text: str) -> str:
    text = strip_qwen3_asr_protocol_text(text)
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")


def split_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    max_chunks: int | None,
) -> list[np.ndarray]:
    chunk_samples = max(1, int(chunk_seconds * sample_rate))
    chunks = [
        audio[start : start + chunk_samples]
        for start in range(0, len(audio), chunk_samples)
    ]
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    return [chunk for chunk in chunks if len(chunk) > 0]


def join_audio(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks)


def select_audio_window(
    chunks: list[np.ndarray],
    sample_rate: int,
    chunk_index: int,
    audio_window_seconds: float,
) -> tuple[np.ndarray, int, int]:
    end_chunk = chunk_index
    if audio_window_seconds <= 0:
        start_chunk = 0
    else:
        window_samples = int(audio_window_seconds * sample_rate)
        samples = 0
        start_chunk = end_chunk
        while start_chunk > 0 and samples < window_samples:
            start_chunk -= 1
            samples += len(chunks[start_chunk])

    window = join_audio(chunks[start_chunk:end_chunk])
    start_sample = sum(len(chunk) for chunk in chunks[:start_chunk])
    end_sample = start_sample + len(window)
    return window, start_sample, end_sample


def build_asr_prompt(
    *,
    engine: AsyncLLM,
    audio: np.ndarray,
    language: str | None,
    context: str,
) -> TokensPrompt:
    tokenizer = cached_tokenizer_from_config(engine.model_config)
    audio_placeholder = Qwen3ASRForConditionalGeneration.get_placeholder_str(
        "audio", 0
    )

    clean_context = sanitize_context(context).strip()
    system_turn = (
        f"<|im_start|>system\n{clean_context}<|im_end|>\n"
        if clean_context
        else ""
    )
    prompt = (
        f"{system_turn}"
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if language is not None:
        full_language = Qwen3ASRForConditionalGeneration.supported_languages.get(
            language, language
        )
        prompt += f"language {full_language}{_ASR_TEXT_TAG}"

    return TokensPrompt(
        prompt_token_ids=tokenizer.encode(prompt),
        multi_modal_data={"audio": audio},
    )


def make_engine_args(args: argparse.Namespace) -> AsyncEngineArgs:
    if not args.enable_general_plugins:
        skip_general_plugins_for_probe()

    hf_overrides: dict[str, Any] = {
        "architectures": ["Qwen3ASRForConditionalGeneration"]
    }
    if args.hf_overrides:
        hf_overrides.update(json.loads(args.hf_overrides))

    return AsyncEngineArgs(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        hf_overrides=hf_overrides,
    )


async def decode_once(
    *,
    engine: AsyncLLM,
    prompt: TokensPrompt,
    sampling_params: SamplingParams,
    request_id: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    raw_parts: list[str] = []
    token_count = 0
    raw_ttft_ms: float | None = None
    ttft_ms: float | None = None
    last_clean_text = ""

    async for output in engine.generate(prompt, sampling_params, request_id):
        now = time.perf_counter()
        for completion in output.outputs:
            text = completion.text or ""
            token_ids = list(completion.token_ids or [])
            token_count += len(token_ids)
            if text:
                if raw_ttft_ms is None:
                    raw_ttft_ms = seconds_to_ms(now - start_time)
                raw_parts.append(text)

            clean_text = strip_qwen3_asr_protocol_text("".join(raw_parts))
            if clean_text.startswith(last_clean_text):
                clean_delta = clean_text[len(last_clean_text) :]
            else:
                clean_delta = clean_text
            if clean_delta and ttft_ms is None:
                ttft_ms = seconds_to_ms(now - start_time)
            last_clean_text = clean_text

    latency_ms = seconds_to_ms(time.perf_counter() - start_time)
    raw_text = "".join(raw_parts)
    return {
        "raw_text": raw_text,
        "clean_text": strip_qwen3_asr_protocol_text(raw_text),
        "latency_ms": latency_ms,
        "raw_ttft_ms": raw_ttft_ms,
        "ttft_ms": ttft_ms,
        "tokens": token_count,
    }


def longest_suffix_prefix_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    for overlap_len in range(max_len, 0, -1):
        if left[-overlap_len:] == right[:overlap_len]:
            return overlap_len
    return 0


def merge_with_overlap(prefix: str, next_text: str) -> tuple[str, int, str]:
    overlap = longest_suffix_prefix_overlap(prefix, next_text)
    delta = next_text[overlap:]
    return prefix + delta, overlap, delta


def split_holdback(
    text: str,
    holdback_words: int,
    holdback_chars: int,
) -> tuple[str, str]:
    if not text:
        return "", ""
    if holdback_words > 0:
        parts = text.split()
        if len(parts) <= holdback_words:
            return "", text
        stable_words = parts[:-holdback_words]
        tail_words = parts[-holdback_words:]
        return " ".join(stable_words), " ".join(tail_words)
    if holdback_chars > 0:
        if len(text) <= holdback_chars:
            return "", text
        return text[:-holdback_chars], text[-holdback_chars:]
    return text, ""


def text_tail(text: str, max_words: int, max_chars: int) -> str:
    if not text:
        return ""
    if max_words > 0:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[-max_words:])
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]
    return text


def normalize_content(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    kept: list[str] = []
    for char in normalized:
        if char in string.whitespace or unicodedata.category(char).startswith("P"):
            continue
        if unicodedata.category(char).startswith("S"):
            continue
        kept.append(char)
    return "".join(kept)


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compare_text(reference: str, hypothesis: str) -> dict[str, Any]:
    import difflib

    ref = normalize_content(reference)
    hyp = normalize_content(hypothesis)
    similarity = difflib.SequenceMatcher(None, ref, hyp).ratio()
    cer = levenshtein_distance(ref, hyp) / max(1, len(ref))
    return {
        "content_similarity": similarity,
        "content_cer": cer,
        "reference_chars": len(reference),
        "hypothesis_chars": len(hypothesis),
        "normalized_reference_chars": len(ref),
        "normalized_hypothesis_chars": len(hyp),
    }


def build_context(tail: str) -> str:
    if not tail:
        return ""
    return (
        "Continue the transcription after this previous transcript tail. "
        f"Previous transcript tail: {tail}"
    )


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio_path)
    audio, sample_rate = load_audio(str(audio_path), sr=args.sample_rate, mono=True)
    chunks = split_audio(audio, sample_rate, args.chunk_seconds, args.max_chunks)
    if not chunks:
        raise ValueError("No audio chunks were produced from --audio-path")

    covered_audio = join_audio(chunks)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        output_kind=RequestOutputKind.DELTA,
    )

    print(
        "[setup] "
        f"audio={audio_path} duration_ms={len(covered_audio) / sample_rate * 1000:.0f} "
        f"sample_rate={sample_rate} chunks={len(chunks)} "
        f"chunk_ms={args.chunk_seconds * 1000:.0f} "
        f"audio_window_ms={args.audio_window_seconds * 1000:.0f} "
        f"model={args.model}",
        flush=True,
    )

    engine = AsyncLLM.from_engine_args(make_engine_args(args))
    rounds: list[dict[str, Any]] = []
    merged_text = ""
    stable_text = ""
    pending_text = ""
    start_time = time.perf_counter()

    try:
        baseline_prompt = build_asr_prompt(
            engine=engine,
            audio=covered_audio,
            language=args.language,
            context="",
        )
        print("[baseline-start]", flush=True)
        baseline = await decode_once(
            engine=engine,
            prompt=baseline_prompt,
            sampling_params=sampling_params,
            request_id=f"{args.request_id}-baseline",
        )
        print(
            "[baseline-done] "
            f"latency_ms={baseline['latency_ms']:.0f} "
            f"ttft_ms={baseline['ttft_ms']}",
            flush=True,
        )

        for chunk_index in range(1, len(chunks) + 1):
            window_audio, start_sample, end_sample = select_audio_window(
                chunks,
                sample_rate,
                chunk_index,
                args.audio_window_seconds,
            )
            tail = text_tail(
                merged_text,
                args.max_tail_words,
                args.max_tail_chars,
            )
            context = build_context(tail)
            prompt = build_asr_prompt(
                engine=engine,
                audio=window_audio,
                language=args.language,
                context=context,
            )
            print(
                "[round-start] "
                f"round={chunk_index} "
                f"audio_ms={start_sample / sample_rate * 1000:.0f}-"
                f"{end_sample / sample_rate * 1000:.0f} "
                f"tail_chars={len(tail)}",
                flush=True,
            )
            decoded = await decode_once(
                engine=engine,
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=f"{args.request_id}-round-{chunk_index}",
            )
            new_merged_text, overlap_chars, delta = merge_with_overlap(
                merged_text,
                str(decoded["clean_text"]),
            )
            stable_text, pending_text = split_holdback(
                new_merged_text,
                args.holdback_words,
                args.holdback_chars,
            )
            duplicate_ratio = overlap_chars / max(1, len(str(decoded["clean_text"])))
            merged_text = new_merged_text
            round_result = {
                "round": chunk_index,
                "audio_start_ms": start_sample / sample_rate * 1000,
                "audio_end_ms": end_sample / sample_rate * 1000,
                "tail_text": tail if args.include_text else None,
                "raw_text": decoded["raw_text"] if args.include_text else None,
                "clean_text": decoded["clean_text"] if args.include_text else None,
                "delta": delta if args.include_text else None,
                "overlap_chars": overlap_chars,
                "delta_chars": len(delta),
                "merged_chars": len(merged_text),
                "stable_chars": len(stable_text),
                "pending_chars": len(pending_text),
                "duplicate_ratio": duplicate_ratio,
                "latency_ms": decoded["latency_ms"],
                "ttft_ms": decoded["ttft_ms"],
                "tokens": decoded["tokens"],
            }
            rounds.append(round_result)
            print(
                "[round-done] "
                f"round={chunk_index} latency_ms={decoded['latency_ms']:.0f} "
                f"ttft_ms={decoded['ttft_ms']} "
                f"overlap_chars={overlap_chars} delta_chars={len(delta)} "
                f"merged_chars={len(merged_text)} "
                f"duplicate_ratio={duplicate_ratio:.3f}",
                flush=True,
            )
            if args.include_text:
                print("[round-clean-text]")
                print(decoded["clean_text"])
                print("[round-delta]")
                print(delta)
    finally:
        engine.shutdown()

    e2e_ms = seconds_to_ms(time.perf_counter() - start_time)
    comparison = compare_text(str(baseline["clean_text"]), merged_text)
    summary = {
        "config": vars(args),
        "audio": {
            "path": str(audio_path.resolve()),
            "duration_ms": len(covered_audio) / sample_rate * 1000,
            "sample_rate": sample_rate,
        },
        "baseline": baseline if args.include_text else {
            key: value for key, value in baseline.items()
            if key not in ("raw_text", "clean_text")
        },
        "rounds": rounds,
        "metrics": {
            "e2e_ms": e2e_ms,
            "rounds": len(rounds),
            "merged_chars": len(merged_text),
            "stable_chars": len(stable_text),
            "pending_chars": len(pending_text),
            **comparison,
        },
        "merged_text": merged_text if args.include_text else None,
        "stable_text": stable_text if args.include_text else None,
        "pending_text": pending_text if args.include_text else None,
    }

    print("[summary]")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False), flush=True)
    if args.include_text:
        print("[baseline-text]")
        print(baseline["clean_text"])
        print("[merged-text]")
        print(merged_text)
        print("[stable-text]")
        print(stable_text)
        print("[pending-text]")
        print(pending_text)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3-ASR semantic streaming wrapper behavior."
    )
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument(
        "--audio-window-seconds",
        type=float,
        default=0.0,
        help="Use cumulative audio when <=0; otherwise use a sliding window.",
    )
    parser.add_argument("--max-tail-words", type=int, default=32)
    parser.add_argument("--max-tail-chars", type=int, default=240)
    parser.add_argument("--holdback-words", type=int, default=5)
    parser.add_argument("--holdback-chars", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-id", default="qwen3-asr-streaming-wrapper-probe")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-general-plugins",
        action="store_true",
        help="Run vLLM general plugins instead of skipping them for this probe.",
    )
    parser.add_argument(
        "--hf-overrides",
        default=None,
        help=(
            "JSON object merged into hf_overrides. The default forces "
            "architectures=['Qwen3ASRForConditionalGeneration']."
        ),
    )
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument(
        "--output-file",
        default="qwen3_asr_streaming_wrapper_probe.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_probe(args))
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
