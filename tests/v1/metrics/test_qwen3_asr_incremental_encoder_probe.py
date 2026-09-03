# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import wave

import numpy as np
import pytest

from examples.speech_to_text.openai.qwen3_asr_incremental_encoder_probe import (
    _as_apply_model_function,
    _build_parser,
    _resolve_modes,
    attention_window_feature_frames,
    attention_sequence_lengths,
    derive_window_geometry,
    encoder_output_length,
    load_audio_cases,
    passes_numerical_gate,
    stable_feature_frames,
    summarize_timings,
    tail_conv_context_dummy_frames,
    tensor_metrics,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_parser_defaults_to_float16_without_quantization():
    args = _build_parser().parse_args(
        ["--model-path", "model", "--output-dir", "output"]
    )

    assert args.dtype == "float16"
    assert args.quantization == "none"
    assert args.load_format == "auto"
    assert not args.enforce_eager
    assert args.attention_window_seconds is None


@pytest.mark.parametrize(("seconds", "frames"), [(4.0, 400), (2.0, 200)])
def test_attention_window_seconds_map_to_feature_frames(seconds, frames):
    assert attention_window_feature_frames(
        seconds, sampling_rate=16000, hop_length=160
    ) == frames


def test_attention_window_seconds_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        attention_window_feature_frames(
            0.0, sampling_rate=16000, hop_length=160
        )


def test_parser_accepts_smaller_attention_window():
    args = _build_parser().parse_args(
        [
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--attention-window-seconds",
            "4",
        ]
    )

    assert args.attention_window_seconds == 4.0


def test_parser_accepts_enforce_eager():
    args = _build_parser().parse_args(
        [
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--enforce-eager",
        ]
    )

    assert args.enforce_eager
    assert _resolve_modes(args.mode, args.enforce_eager) == ("eager",)


def test_parser_accepts_one_audio_file():
    args = _build_parser().parse_args(
        [
            "--model-path",
            "model",
            "--output-dir",
            "output",
            "--audio-file",
            "sample.wav",
        ]
    )

    assert str(args.audio_file) == "sample.wav"
    assert args.audio_manifest is None


def test_parser_rejects_audio_file_with_manifest():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "--model-path",
                "model",
                "--output-dir",
                "output",
                "--audio-file",
                "sample.wav",
                "--audio-manifest",
                "manifest.jsonl",
            ]
        )


def test_single_audio_file_does_not_add_synthetic_cases(tmp_path):
    audio_path = tmp_path / "sample.wav"
    samples = np.asarray([0, 16384, -16384], dtype="<i2")
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())

    cases = load_audio_cases(
        audio_file=audio_path,
        audio_manifest=None,
        sampling_rate=16000,
        skip_synthetic=False,
    )

    assert len(cases) == 1
    assert cases[0].case_id == "sample"
    np.testing.assert_allclose(cases[0].audio, [0.0, 0.5, -0.5])


def test_apply_model_wrapper_survives_engine_serialization(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    class Callback:
        def __call__(self, model):
            return {"model": model}

    wrapped = _as_apply_model_function(Callback())
    decoded = MsgpackDecoder().decode(MsgpackEncoder().encode(wrapped))

    assert callable(decoded)
    assert decoded("audio-model") == {"model": "audio-model"}


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


def test_short_incremental_tail_requests_full_conv_padding_context():
    assert tail_conv_context_dummy_frames(0, 100) == 0
    assert tail_conv_context_dummy_frames(50, 100) == 100
    assert tail_conv_context_dummy_frames(99, 100) == 100
    assert tail_conv_context_dummy_frames(100, 100) == 0
    assert tail_conv_context_dummy_frames(250, 100) == 0


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
