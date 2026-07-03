# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe whether Qwen3-ASR audio features can be cached by chunk.

This script intentionally does not benchmark pseudo-streaming transcript merge.
It focuses on the lower-level question needed for Engine-level streaming:
whether processing a full audio clip is equivalent to processing audio chunks
and concatenating their extracted features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoProcessor

from vllm.config import ModelConfig
from vllm.multimodal.media.audio import load_audio
from vllm.transformers_utils.processor import cached_processor_from_config


def qwen3_asr_audio_token_len(input_length: int) -> int:
    """Mirror Qwen3-ASR's audio token length formula."""
    input_lengths_leave = input_length % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2
        + 1
        + (input_length // 100) * 13
    )


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


def parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_qwen3_asr_feature_extractor(args: argparse.Namespace) -> Any:
    processor_path = args.processor or args.model
    processor = None
    load_errors: list[str] = []

    if args.use_vllm_processor:
        try:
            model_config = ModelConfig(
                model=args.model,
                tokenizer=args.tokenizer or args.model,
                trust_remote_code=args.trust_remote_code,
                hf_overrides={
                    "architectures": ["Qwen3ASRForConditionalGeneration"],
                },
            )
            processor = cached_processor_from_config(model_config)
        except Exception as exc:
            load_errors.append(f"vLLM processor load failed: {exc!r}")

    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception as exc:
            load_errors.append(f"AutoProcessor load failed: {exc!r}")
            raise RuntimeError("; ".join(load_errors)) from exc

    if not hasattr(processor, "feature_extractor"):
        detail = "; ".join(load_errors)
        if detail:
            detail = f" Previous load errors: {detail}"
        raise TypeError(
            f"Processor loaded from {processor_path!r} has no feature_extractor."
            f"{detail}"
        )
    return processor.feature_extractor


def extract_features(
    feature_extractor: Any,
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    outputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="np",
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    input_features = as_numpy(outputs["input_features"])
    attention_key = (
        "feature_attention_mask"
        if "feature_attention_mask" in outputs
        else "attention_mask"
    )
    feature_attention_mask = as_numpy(outputs[attention_key])

    if input_features.ndim != 3:
        raise ValueError(
            f"Expected input_features rank 3, got {input_features.shape}"
        )
    if feature_attention_mask.ndim != 2:
        raise ValueError(
            "Expected feature attention mask rank 2, got "
            f"{feature_attention_mask.shape}"
        )

    valid_frames = int(feature_attention_mask[0].sum())
    valid_features = input_features[0, :, :valid_frames]
    return {
        "elapsed_ms": elapsed_ms,
        "input_features_shape": list(input_features.shape),
        "feature_attention_mask_shape": list(feature_attention_mask.shape),
        "valid_frames": valid_frames,
        "audio_tokens": qwen3_asr_audio_token_len(valid_frames),
        "valid_features": valid_features,
    }


def feature_stats(features: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(features.shape),
        "mean": float(np.mean(features)) if features.size else 0.0,
        "std": float(np.std(features)) if features.size else 0.0,
        "min": float(np.min(features)) if features.size else 0.0,
        "max": float(np.max(features)) if features.size else 0.0,
    }


def boundary_keep_mask(
    total_frames: int,
    chunk_frame_lengths: list[int],
    ignore_boundary_frames: int,
) -> np.ndarray:
    mask = np.ones(total_frames, dtype=bool)
    if ignore_boundary_frames <= 0:
        return mask

    cursor = 0
    for length in chunk_frame_lengths[:-1]:
        cursor += length
        start = max(0, cursor - ignore_boundary_frames)
        end = min(total_frames, cursor + ignore_boundary_frames)
        mask[start:end] = False
    return mask


def compare_features(
    full_features: np.ndarray,
    concat_features: np.ndarray,
    chunk_frame_lengths: list[int],
    ignore_boundary_frames: int,
) -> dict[str, Any]:
    common_frames = min(full_features.shape[-1], concat_features.shape[-1])
    if common_frames == 0:
        return {
            "common_frames": 0,
            "mae": None,
            "rmse": None,
            "max_abs": None,
            "interior_frames": 0,
            "interior_mae": None,
            "interior_rmse": None,
            "interior_max_abs": None,
        }

    full_common = full_features[:, :common_frames]
    concat_common = concat_features[:, :common_frames]
    diff = full_common - concat_common

    keep_mask = boundary_keep_mask(
        common_frames,
        chunk_frame_lengths,
        ignore_boundary_frames,
    )
    interior_diff = diff[:, keep_mask]
    result = {
        "common_frames": common_frames,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(float(np.mean(diff * diff)))),
        "max_abs": float(np.max(np.abs(diff))),
        "interior_frames": int(interior_diff.shape[-1]),
        "interior_mae": None,
        "interior_rmse": None,
        "interior_max_abs": None,
    }
    if interior_diff.size:
        result.update(
            {
                "interior_mae": float(np.mean(np.abs(interior_diff))),
                "interior_rmse": float(
                    math.sqrt(float(np.mean(interior_diff * interior_diff)))
                ),
                "interior_max_abs": float(np.max(np.abs(interior_diff))),
            }
        )
    return result


def get_feature_hop_length(feature_extractor: Any, sample_rate: int) -> int:
    hop_length = getattr(feature_extractor, "hop_length", None)
    if hop_length is not None:
        return int(hop_length)
    return sample_rate // 100


def build_overlap_concat_features(
    *,
    feature_extractor: Any,
    audio: np.ndarray,
    chunks: list[np.ndarray],
    sample_rate: int,
    overlap_seconds: float,
    chunk_frame_lengths: list[int],
) -> dict[str, Any]:
    hop_length = get_feature_hop_length(feature_extractor, sample_rate)
    overlap_samples = max(0, int(overlap_seconds * sample_rate))
    cropped_features = []
    cropped_frame_lengths = []
    items = []
    core_start_sample = 0

    for index, chunk in enumerate(chunks, start=1):
        core_end_sample = core_start_sample + len(chunk)
        ext_start_sample = max(0, core_start_sample - overlap_samples)
        ext_end_sample = min(len(audio), core_end_sample + overlap_samples)
        ext_audio = audio[ext_start_sample:ext_end_sample]
        ext_result = extract_features(feature_extractor, ext_audio, sample_rate)

        crop_start_frame = round((core_start_sample - ext_start_sample) / hop_length)
        expected_frames = chunk_frame_lengths[index - 1]
        crop_end_frame = crop_start_frame + expected_frames
        ext_features = ext_result["valid_features"]
        clipped = crop_end_frame > ext_features.shape[-1]
        cropped = ext_features[
            :,
            crop_start_frame : min(crop_end_frame, ext_features.shape[-1]),
        ]

        cropped_features.append(cropped)
        cropped_frame_lengths.append(int(cropped.shape[-1]))
        items.append(
            {
                "index": index,
                "core_audio_start_ms": core_start_sample / sample_rate * 1000,
                "core_audio_end_ms": core_end_sample / sample_rate * 1000,
                "extended_audio_start_ms": ext_start_sample / sample_rate * 1000,
                "extended_audio_end_ms": ext_end_sample / sample_rate * 1000,
                "extract_ms": ext_result["elapsed_ms"],
                "extended_valid_frames": ext_result["valid_frames"],
                "crop_start_frame": crop_start_frame,
                "expected_core_frames": expected_frames,
                "cropped_frames": int(cropped.shape[-1]),
                "crop_clipped": clipped,
            }
        )
        core_start_sample = core_end_sample

    concat_features = (
        np.concatenate(cropped_features, axis=-1)
        if cropped_features
        else np.empty((0, 0), dtype=np.float32)
    )
    concat_frames = int(concat_features.shape[-1])
    return {
        "overlap_seconds": overlap_seconds,
        "overlap_ms": overlap_seconds * 1000,
        "overlap_samples": overlap_samples,
        "hop_length": hop_length,
        "items": items,
        "concat_features": concat_features,
        "concat_valid_frames": concat_frames,
        "concat_as_single_audio_tokens": qwen3_asr_audio_token_len(concat_frames),
        "sum_cropped_audio_tokens": sum(
            qwen3_asr_audio_token_len(length)
            for length in cropped_frame_lengths
        ),
        "cropped_frame_lengths": cropped_frame_lengths,
        "stats": feature_stats(concat_features),
    }


def build_trace() -> list[str]:
    return [
        "raw waveform",
        "Qwen3ASRProcessor / WhisperFeatureExtractor",
        "input_features + feature_attention_mask",
        "audio_feature_lengths = feature_attention_mask.sum(-1)",
        "PromptReplacement repeats <|audio_pad|> by audio token length",
        "Qwen3ASRForConditionalGeneration._process_audio_input",
        "Qwen3OmniMoeAudioEncoder(input_features, audio_feature_lengths)",
        "_merge_multimodal_embeddings inserts audio embeddings into prompt",
        "Qwen3ForCausalLM decodes transcript tokens",
    ]


def build_verdict(
    *,
    full_frames: int,
    concat_frames: int,
    full_audio_tokens: int,
    concat_as_single_audio_tokens: int,
    sum_chunk_audio_tokens: int,
    comparison: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    frames_match = full_frames == concat_frames
    token_lengths_match = full_audio_tokens == concat_as_single_audio_tokens
    mae = comparison.get("interior_mae")
    features_close = mae is not None and mae <= tolerance
    naive_chunk_tokens_match = full_audio_tokens == sum_chunk_audio_tokens

    if frames_match and token_lengths_match and features_close:
        summary = "plausible: concatenated chunk features match full features"
    else:
        summary = (
            "unsafe: chunk feature concatenation differs from full-audio "
            "feature extraction"
        )

    return {
        "summary": summary,
        "frames_match": frames_match,
        "concat_as_single_audio_tokens_match": token_lengths_match,
        "sum_chunk_audio_tokens_match": naive_chunk_tokens_match,
        "features_close_within_tolerance": features_close,
        "tolerance": tolerance,
        "note": (
            "If sum_chunk_audio_tokens differs from full_audio_tokens, caching "
            "post-audio-encoder embeddings per chunk would change sequence "
            "lengths. If concat feature values differ, raw feature caching needs "
            "boundary overlap or a rolling feature extractor state."
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio_path)
    audio, sample_rate = load_audio(str(audio_path), sr=args.sample_rate, mono=True)
    chunks = split_audio(audio, sample_rate, args.chunk_seconds, args.max_chunks)
    covered_audio = join_audio(chunks)
    if len(covered_audio) == 0:
        raise ValueError("No audio was selected for probing")

    feature_extractor = load_qwen3_asr_feature_extractor(args)
    full = extract_features(feature_extractor, covered_audio, sample_rate)

    chunk_results = []
    chunk_features = []
    chunk_frame_lengths = []
    start_sample = 0
    for index, chunk in enumerate(chunks, start=1):
        chunk_result = extract_features(feature_extractor, chunk, sample_rate)
        end_sample = start_sample + len(chunk)
        chunk_frame_lengths.append(int(chunk_result["valid_frames"]))
        chunk_features.append(chunk_result["valid_features"])
        chunk_results.append(
            {
                "index": index,
                "audio_start_ms": start_sample / sample_rate * 1000,
                "audio_end_ms": end_sample / sample_rate * 1000,
                "samples": len(chunk),
                "extract_ms": chunk_result["elapsed_ms"],
                "valid_frames": chunk_result["valid_frames"],
                "audio_tokens": chunk_result["audio_tokens"],
                "input_features_shape": chunk_result["input_features_shape"],
                "feature_attention_mask_shape": (
                    chunk_result["feature_attention_mask_shape"]
                ),
            }
        )
        start_sample = end_sample

    concat_features = (
        np.concatenate(chunk_features, axis=-1)
        if chunk_features
        else np.empty((0, 0), dtype=np.float32)
    )
    concat_frames = int(concat_features.shape[-1])
    concat_as_single_audio_tokens = qwen3_asr_audio_token_len(concat_frames)
    sum_chunk_audio_tokens = sum(int(item["audio_tokens"]) for item in chunk_results)
    comparison = compare_features(
        full["valid_features"],
        concat_features,
        chunk_frame_lengths,
        args.ignore_boundary_frames,
    )
    verdict = build_verdict(
        full_frames=int(full["valid_frames"]),
        concat_frames=concat_frames,
        full_audio_tokens=int(full["audio_tokens"]),
        concat_as_single_audio_tokens=concat_as_single_audio_tokens,
        sum_chunk_audio_tokens=sum_chunk_audio_tokens,
        comparison=comparison,
        tolerance=args.feature_tolerance,
    )
    overlap_comparisons = []
    for overlap_seconds in parse_float_list(args.feature_overlap_seconds):
        overlap_result = build_overlap_concat_features(
            feature_extractor=feature_extractor,
            audio=covered_audio,
            chunks=chunks,
            sample_rate=sample_rate,
            overlap_seconds=overlap_seconds,
            chunk_frame_lengths=chunk_frame_lengths,
        )
        overlap_comparison = compare_features(
            full["valid_features"],
            overlap_result["concat_features"],
            overlap_result["cropped_frame_lengths"],
            args.ignore_boundary_frames,
        )
        overlap_verdict = build_verdict(
            full_frames=int(full["valid_frames"]),
            concat_frames=int(overlap_result["concat_valid_frames"]),
            full_audio_tokens=int(full["audio_tokens"]),
            concat_as_single_audio_tokens=int(
                overlap_result["concat_as_single_audio_tokens"]
            ),
            sum_chunk_audio_tokens=int(overlap_result["sum_cropped_audio_tokens"]),
            comparison=overlap_comparison,
            tolerance=args.feature_tolerance,
        )
        overlap_comparisons.append(
            {
                key: value
                for key, value in overlap_result.items()
                if key != "concat_features"
            }
            | {
                "feature_comparison": overlap_comparison,
                "verdict": overlap_verdict,
            }
        )

    summary = {
        "config": vars(args),
        "audio": {
            "path": str(audio_path.resolve()),
            "sample_rate": sample_rate,
            "covered_samples": len(covered_audio),
            "covered_ms": len(covered_audio) / sample_rate * 1000,
            "chunks": len(chunks),
            "chunk_ms": args.chunk_seconds * 1000,
        },
        "processor": {
            "feature_extractor": type(feature_extractor).__name__,
            "sampling_rate": getattr(feature_extractor, "sampling_rate", None),
            "hop_length": getattr(feature_extractor, "hop_length", None),
            "chunk_length": getattr(feature_extractor, "chunk_length", None),
        },
        "full_audio_features": {
            "extract_ms": full["elapsed_ms"],
            "valid_frames": full["valid_frames"],
            "audio_tokens": full["audio_tokens"],
            "input_features_shape": full["input_features_shape"],
            "feature_attention_mask_shape": full[
                "feature_attention_mask_shape"
            ],
            "stats": feature_stats(full["valid_features"]),
        },
        "chunk_audio_features": {
            "items": chunk_results,
            "concat_valid_frames": concat_frames,
            "concat_as_single_audio_tokens": concat_as_single_audio_tokens,
            "sum_chunk_audio_tokens": sum_chunk_audio_tokens,
            "stats": feature_stats(concat_features),
        },
        "feature_comparison": comparison,
        "overlap_feature_comparisons": overlap_comparisons,
        "processor_path_trace": build_trace(),
        "verdict": verdict,
    }

    print("[setup]")
    print(
        f"audio={audio_path} covered_ms={summary['audio']['covered_ms']:.0f} "
        f"chunks={len(chunks)} chunk_ms={args.chunk_seconds * 1000:.0f} "
        f"model={args.model}"
    )
    print("[processor]")
    print(json.dumps(summary["processor"], indent=2, ensure_ascii=False))
    print("[full-audio-features]")
    print(
        json.dumps(
            summary["full_audio_features"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("[chunk-audio-features]")
    print(
        json.dumps(
            summary["chunk_audio_features"],
            indent=2,
            ensure_ascii=False,
        )
    )
    print("[feature-comparison]")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    if overlap_comparisons:
        print("[overlap-feature-comparisons]")
        print(json.dumps(overlap_comparisons, indent=2, ensure_ascii=False))
    print("[processor-path-trace]")
    for step in summary["processor_path_trace"]:
        print(f"- {step}")
    print("[verdict]")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Qwen3-ASR chunk feature cache feasibility."
    )
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument("--processor", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--max-chunks", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-vllm-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load the Qwen3-ASR processor through vLLM ModelConfig first. "
            "This is needed when AutoProcessor does not expose "
            "feature_extractor for local Qwen3-ASR checkpoints."
        ),
    )
    parser.add_argument("--ignore-boundary-frames", type=int, default=2)
    parser.add_argument(
        "--feature-overlap-seconds",
        default="0.5,1.0,2.0",
        help=(
            "Comma-separated overlap sizes. For each value, the probe extracts "
            "features from chunk audio plus left/right context, crops the core "
            "chunk frames, concatenates them, and compares against full audio."
        ),
    )
    parser.add_argument("--feature-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--output-file",
        default="qwen3_asr_feature_cache_probe.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_probe(args)
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
