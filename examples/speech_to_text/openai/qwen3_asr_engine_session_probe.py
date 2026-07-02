# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe Qwen3-ASR Engine-level streaming sessions.

This script uses the local vLLM V1 AsyncLLM streaming-input path instead of the
OpenAI REST audio endpoint. It is meant to validate the existing resumable
request/session mechanics before making deeper scheduler or KV-cache changes.

The current Qwen3ASRRealtimeGeneration path appends one audio prompt per chunk
to the same Engine request. That exercises Engine session KV retention, but it
is still a diagnostic path rather than the final ideal ASR streaming design.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm import SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.model_executor.models.qwen3_asr import (
    _ASR_TEXT_TAG,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.qwen3_asr_realtime import (
    Qwen3ASRRealtimeGeneration,
)
from vllm.multimodal.media.audio import load_audio
from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.engine.async_llm import AsyncLLM


_PROTOCOL_PREFIX_RE = re.compile(
    rf"(?:^|\s*)language\s+[^<\r\n]{{1,80}}{re.escape(_ASR_TEXT_TAG)}",
    flags=re.IGNORECASE,
)


def skip_general_plugins_for_probe() -> None:
    # This local probe inspects Engine streaming mechanics. Some out-of-tree
    # general plugins patch unrelated modules and can fail before the probe
    # starts when plugin and checkout versions drift. Keep platform plugins
    # untouched, but skip general plugin hooks unless explicitly requested.
    import vllm.engine.arg_utils as arg_utils
    import vllm.plugins as plugins

    def load_no_general_plugins() -> None:
        plugins.plugins_loaded = True

    plugins.load_general_plugins = load_no_general_plugins
    arg_utils.load_general_plugins = load_no_general_plugins


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def approximate_feature_frames(num_samples: int, sample_rate: int) -> int:
    # Qwen3-ASR uses Whisper-style audio features, roughly 100 frames / second.
    return int(np.ceil(num_samples / sample_rate * 100.0))


def estimate_qwen3_asr_audio_tokens(num_samples: int, sample_rate: int) -> int:
    frames = approximate_feature_frames(num_samples, sample_rate)
    return int(_get_feat_extract_output_lengths(torch.tensor(frames)).item())


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


def build_realtime_prompt(
    *,
    engine: AsyncLLM,
    audio_chunk: np.ndarray,
    chunk_index: int,
    language: str | None,
    prompt_mode: str,
) -> tuple[TokensPrompt, dict[str, Any]]:
    tokenizer = cached_tokenizer_from_config(engine.model_config)
    audio_placeholder = Qwen3ASRRealtimeGeneration.get_placeholder_str("audio", 0)

    use_initial_prompt = chunk_index == 1 or prompt_mode == "full"
    if use_initial_prompt:
        prompt = (
            f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if language is not None:
            full_language = Qwen3ASRRealtimeGeneration.supported_languages.get(
                language, language
            )
            prompt += f"language {full_language}{_ASR_TEXT_TAG}"
    elif prompt_mode == "audio_only":
        prompt = audio_placeholder
    elif prompt_mode == "audio_with_boundary":
        prompt = (
            f"<|im_end|>\n<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        raise ValueError(f"Unsupported --chunk-prompt-mode: {prompt_mode}")

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_info = {
        "prompt_mode": prompt_mode,
        "prompt_chars": len(prompt),
        "prompt_tokens": len(prompt_token_ids),
        "uses_initial_prompt": use_initial_prompt,
        "has_language_prefix": use_initial_prompt and language is not None,
        "has_audio_placeholder": audio_placeholder in prompt,
    }
    return TokensPrompt(
        prompt_token_ids=prompt_token_ids,
        multi_modal_data={"audio": audio_chunk},
    ), prompt_info


def strip_qwen3_asr_protocol_text(text: str) -> str:
    """Remove repeated Qwen3-ASR protocol prefixes without assuming language."""
    if not text:
        return ""

    text = _PROTOCOL_PREFIX_RE.sub("", text)
    return text.replace(_ASR_TEXT_TAG, "")


def make_engine_args(args: argparse.Namespace) -> AsyncEngineArgs:
    if not args.enable_general_plugins:
        skip_general_plugins_for_probe()

    hf_overrides: dict[str, Any] = {"architectures": ["Qwen3ASRRealtimeGeneration"]}
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


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio_path)
    audio, sample_rate = load_audio(str(audio_path), sr=args.sample_rate, mono=True)
    chunks = split_audio(audio, sample_rate, args.chunk_seconds, args.max_chunks)
    if not chunks:
        raise ValueError("No audio chunks were produced from --audio-path")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        output_kind=RequestOutputKind.DELTA,
    )

    print(
        "[setup] "
        f"audio={audio_path} duration_ms={len(audio) / sample_rate * 1000.0:.0f} "
        f"sample_rate={sample_rate} chunks={len(chunks)} "
        f"chunk_ms={args.chunk_seconds * 1000.0:.0f} "
        f"model={args.model} prompt_mode={args.chunk_prompt_mode}",
        flush=True,
    )
    print(
        "[engine] architecture_override=Qwen3ASRRealtimeGeneration "
        "path=AsyncLLM.StreamingInput resumable_request=true "
        "platform_plugins=enabled "
        f"general_plugins={'enabled' if args.enable_general_plugins else 'skipped'}",
        flush=True,
    )

    engine = AsyncLLM.from_engine_args(make_engine_args(args))
    chunk_stats: list[dict[str, Any]] = []
    output_events: list[dict[str, Any]] = []
    all_text_parts: list[str] = []
    last_clean_text = ""
    raw_ttft_ms: float | None = None
    ttft_ms: float | None = None
    start_time = time.perf_counter()

    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        cumulative_samples = 0
        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_start_ms = cumulative_samples / sample_rate * 1000.0
            cumulative_samples += len(chunk)
            chunk_end_ms = cumulative_samples / sample_rate * 1000.0
            audio_tokens = estimate_qwen3_asr_audio_tokens(len(chunk), sample_rate)
            prompt, prompt_info = build_realtime_prompt(
                engine=engine,
                audio_chunk=chunk,
                chunk_index=chunk_idx,
                language=args.language,
                prompt_mode=args.chunk_prompt_mode,
            )
            chunk_stat = {
                "chunk_index": chunk_idx,
                "audio_start_ms": chunk_start_ms,
                "audio_end_ms": chunk_end_ms,
                "samples": len(chunk),
                "approx_audio_tokens": audio_tokens,
                **prompt_info,
            }
            chunk_stats.append(chunk_stat)
            print(
                "[input-chunk] "
                f"chunk={chunk_idx} audio_ms={chunk_start_ms:.0f}-{chunk_end_ms:.0f} "
                f"samples={len(chunk)} approx_audio_tokens={audio_tokens} "
                f"prompt_tokens={prompt_info['prompt_tokens']} "
                f"initial_prompt={prompt_info['uses_initial_prompt']} "
                f"language_prefix={prompt_info['has_language_prefix']}",
                flush=True,
            )
            yield StreamingInput(
                prompt=prompt,
                sampling_params=sampling_params,
            )
            if args.input_delay_ms > 0:
                await asyncio.sleep(args.input_delay_ms / 1000.0)

    try:
        async for output in engine.generate(
            input_generator(),
            sampling_params,
            request_id=args.request_id,
        ):
            now = time.perf_counter()
            for completion in output.outputs:
                text = completion.text or ""
                token_ids = list(completion.token_ids or [])
                if text:
                    if raw_ttft_ms is None:
                        raw_ttft_ms = seconds_to_ms(now - start_time)
                    all_text_parts.append(text)

                clean_text_so_far = strip_qwen3_asr_protocol_text(
                    "".join(all_text_parts)
                )
                if clean_text_so_far.startswith(last_clean_text):
                    clean_delta = clean_text_so_far[len(last_clean_text) :]
                else:
                    clean_delta = clean_text_so_far
                if clean_delta and ttft_ms is None:
                    ttft_ms = seconds_to_ms(now - start_time)
                last_clean_text = clean_text_so_far

                event = {
                    "elapsed_ms": seconds_to_ms(now - start_time),
                    "finished": output.finished,
                    "text": clean_delta if args.include_text else None,
                    "raw_text": text if args.include_text else None,
                    "clean_text": clean_delta if args.include_text else None,
                    "num_token_ids": len(token_ids),
                }
                output_events.append(event)
                print(
                    "[output] "
                    f"elapsed_ms={event['elapsed_ms']:.0f} "
                    f"finished={output.finished} "
                    f"raw_delta_chars={len(text)} "
                    f"clean_delta_chars={len(clean_delta)} "
                    f"delta_tokens={len(token_ids)}",
                    flush=True,
                )
                if args.include_text and text:
                    print("[output-raw-text]")
                    print(text)
                    if clean_delta:
                        print("[output-clean-text]")
                        print(clean_delta)

        e2e_ms = seconds_to_ms(time.perf_counter() - start_time)
    finally:
        engine.shutdown()

    final_raw_text = "".join(all_text_parts)
    final_clean_text = strip_qwen3_asr_protocol_text(final_raw_text)
    cumulative_audio_tokens = sum(
        int(chunk["approx_audio_tokens"]) for chunk in chunk_stats
    )
    reusable_audio_tokens_after_first = max(
        0,
        cumulative_audio_tokens
        - int(chunk_stats[-1]["approx_audio_tokens"]),
    )

    summary = {
        "config": vars(args),
        "audio": {
            "path": str(audio_path.resolve()),
            "duration_ms": len(audio) / sample_rate * 1000.0,
            "sample_rate": sample_rate,
        },
        "chunks": chunk_stats,
        "metrics": {
            "ttft_ms": ttft_ms,
            "raw_ttft_ms": raw_ttft_ms,
            "e2e_ms": e2e_ms,
            "output_events": len(output_events),
            "final_chars": len(final_clean_text),
            "final_words": len(final_clean_text.split()),
            "raw_final_chars": len(final_raw_text),
            "raw_final_words": len(final_raw_text.split()),
            "protocol_tag_count": final_raw_text.count(_ASR_TEXT_TAG),
            "protocol_prefix_count": len(
                _PROTOCOL_PREFIX_RE.findall(final_raw_text)
            ),
        },
        "cache_probe": {
            "engine_path": "AsyncLLM StreamingInput resumable request",
            "approx_total_audio_tokens": cumulative_audio_tokens,
            "approx_reusable_audio_tokens_before_last_chunk": (
                reusable_audio_tokens_after_first
            ),
            "transcript_kv_reuse_assumption": (
                "not reused safely; generated text is folded back as prompt "
                "tokens by the resumable session path"
            ),
        },
        "outputs": output_events,
        "final_text": final_clean_text if args.include_text else None,
        "final_raw_text": final_raw_text if args.include_text else None,
    }

    print("[summary]")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False), flush=True)
    print("[cache-probe]")
    print(json.dumps(summary["cache_probe"], indent=2, ensure_ascii=False), flush=True)
    if args.include_text:
        print("[final-raw-text]")
        print(final_raw_text)
        print("[final-clean-text]")
        print(final_clean_text)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3-ASR local Engine streaming session behavior."
    )
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument(
        "--chunk-prompt-mode",
        choices=("full", "audio_only", "audio_with_boundary"),
        default="full",
        help=(
            "Prompt shape for chunks after the first chunk. 'full' repeats the "
            "current full ChatML audio prompt for every chunk. 'audio_only' "
            "appends only the audio placeholder after the first chunk. "
            "'audio_with_boundary' closes the prior assistant turn and opens a "
            "new user-audio/assistant turn without repeating the language tag."
        ),
    )
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-id", default="qwen3-asr-engine-session-probe")
    parser.add_argument("--input-delay-ms", type=float, default=0.0)
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
            "JSON object merged into the default hf_overrides. The default "
            "forces architectures=['Qwen3ASRRealtimeGeneration']."
        ),
    )
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument(
        "--output-file",
        default="qwen3_asr_engine_session_probe.json",
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
