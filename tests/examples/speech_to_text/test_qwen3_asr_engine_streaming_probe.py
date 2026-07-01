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
    assert analysis["candidate_decoder_kv_reuse"]["fixed_text_prompt_kv"][
        "safe"
    ]
    assert analysis["candidate_decoder_kv_reuse"]["prefix_audio_kv"][
        "safe_if_audio_embeddings_are_stable"
    ]
    assert analysis["candidate_decoder_kv_reuse"]["new_audio_kv"][
        "must_compute"
    ]
    assert not analysis["candidate_decoder_kv_reuse"][
        "assistant_transcript_kv"
    ]["safe"]
    assert "unsafe" in analysis["verdict"]


def test_content_similarity_ignores_punctuation_and_spaces():
    left = "Hello, world! This is a test."
    right = "hello world this is a test"

    assert probe.content_similarity(left, right) == 1.0
    assert probe.content_cer(left, right) == 0.0


def test_merge_text_with_content_overlap_trims_duplicate_boundary():
    left = "Hello, world. This is the first chunk."
    right = "this is the first chunk. And this is the continuation."

    merged, overlap_chars = probe.merge_text_with_content_overlap(
        left,
        right,
        min_overlap_chars=10,
    )

    assert overlap_chars > 0
    assert merged == (
        "Hello, world. This is the first chunk. "
        "And this is the continuation."
    )


def test_merge_text_with_content_overlap_keeps_non_overlapping_text():
    merged, overlap_chars = probe.merge_text_with_content_overlap(
        "prefix text",
        "new continuation",
        min_overlap_chars=10,
    )

    assert overlap_chars == 0
    assert merged == "prefix text new continuation"
