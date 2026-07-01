# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from examples.speech_to_text.openai import (
    qwen3_asr_engine_streaming_probe as probe,
)


def test_decoder_kv_reuse_is_unsafe_when_audio_grows():
    analysis = probe.analyze_decoder_kv_reuse(
        prefix_samples=16000 * 10,
        target_samples=16000 * 20,
        sample_rate=16000,
    )

    assert analysis["inserted_audio_tokens_before_assistant"] > 0
    assert analysis["naive_decoder_kv_reuse_safe"] is False
    assert "unsafe" in analysis["verdict"]


def test_content_similarity_ignores_punctuation_and_spaces():
    left = "Hello, world! This is a test."
    right = "hello world this is a test"

    assert probe.content_similarity(left, right) == 1.0
    assert probe.content_cer(left, right) == 0.0

