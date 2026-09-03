# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from examples.speech_to_text.openai.qwen3_asr_incremental_encoder_probe import (
    _build_parser,
    attention_sequence_lengths,
    derive_window_geometry,
    encoder_output_length,
    passes_numerical_gate,
    stable_feature_frames,
    summarize_timings,
    tensor_metrics,
)


def test_parser_defaults_to_float16_without_quantization():
    args = _build_parser().parse_args(
        ["--model-path", "model", "--output-dir", "output"]
    )

    assert args.dtype == "float16"
    assert args.quantization == "none"
    assert args.load_format == "auto"


def test_window_geometry_is_derived_from_runtime_values():
    geometry = derive_window_geometry(
        n_window=50,
        n_window_infer=400,
        conv_chunksize=500,
        sampling_rate=16000,
        hop_length=160,
    )

    assert geometry.conv_feature_window == 100
    assert geometry.attention_feature_window == 400
    assert geometry.conv_chunks_per_attention_window == 4
    assert geometry.encoder_tokens_per_attention_window == 52
    assert geometry.attention_window_seconds == 4.0
    assert attention_sequence_lengths(400, geometry) == [52]
    assert attention_sequence_lengths(800, geometry) == [52, 52]


@pytest.mark.parametrize(
    ("feature_frames", "expected"),
    [(0, 0), (399, 0), (400, 400), (799, 400), (800, 800)],
)
def test_stable_feature_frames_only_returns_complete_attention_windows(
    feature_frames, expected
):
    assert stable_feature_frames(feature_frames, 400) == expected


def test_invalid_window_ratio_is_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        derive_window_geometry(
            n_window=100,
            n_window_infer=350,
            conv_chunksize=500,
            sampling_rate=16000,
            hop_length=160,
        )


def test_qwen3_asr_output_lengths_at_window_boundaries():
    assert encoder_output_length(100) == 13
    assert encoder_output_length(200) == 26
    assert encoder_output_length(400) == 52
    assert encoder_output_length(800) == 104


def test_tensor_metrics_detects_shape_and_value_mismatch():
    reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
    close = reference + 1e-4

    assert tensor_metrics(reference, close)["allclose"]
    assert not tensor_metrics(reference, np.ones((2, 2)))["shape_equal"]
    assert not tensor_metrics(reference, reference + 1.0)["allclose"]


def test_numerical_gate_uses_allclose_cosine_and_repeat_noise():
    reference = np.asarray([1.0, 2.0], dtype=np.float32)
    close = tensor_metrics(reference, reference + 1e-4)
    noise = [tensor_metrics(reference, reference + 1e-4)]

    assert passes_numerical_gate(close, noise)
    assert not passes_numerical_gate(
        tensor_metrics(reference, reference + 1.0), noise
    )


def test_timing_summary_handles_empty_and_populated_rows():
    assert summarize_timings([])["full_encoder_ms"]["p95"] is None
    summary = summarize_timings(
        [
            {
                "full_encoder_ms": 10.0,
                "incremental_encoder_ms": 6.0,
                "compute_reduction_ratio": 0.4,
            },
            {
                "full_encoder_ms": 20.0,
                "incremental_encoder_ms": 10.0,
                "compute_reduction_ratio": 0.5,
            },
        ]
    )
    assert summary["full_encoder_ms"]["p50"] == 15.0
    assert summary["incremental_encoder_ms"]["p50"] == 8.0
    assert summary["compute_reduction_ratio"]["p50"] == pytest.approx(0.45)
