# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRAudioEmbeddingInputs,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor,
    _qwen3asr_field_config,
)
from vllm.multimodal.parse import AudioEmbeddingItems, MultiModalDataItems


def test_qwen3_asr_precomputed_audio_embeddings_bypass_encoder():
    embeddings = [torch.randn(3, 8), torch.randn(5, 8)]
    audio_input = Qwen3ASRForConditionalGeneration._parse_and_validate_audio_input(
        None, audio_embeds=embeddings
    )

    assert isinstance(audio_input, Qwen3ASRAudioEmbeddingInputs)
    outputs = Qwen3ASRForConditionalGeneration._process_audio_input(
        None, audio_input
    )
    assert len(outputs) == len(embeddings)
    assert all(output is embedding for output, embedding in zip(outputs, embeddings))


def test_qwen3_asr_embedding_prompt_length_matches_embedding_tokens():
    embeddings = [torch.randn(3, 8)]
    mm_items = MultiModalDataItems(
        {"audio": AudioEmbeddingItems(embeddings, expected_hidden_size=8)}
    )
    info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: SimpleNamespace(audio_token="<audio>"),
        get_tokenizer=lambda: SimpleNamespace(
            get_vocab=lambda: {"<audio>": 7}
        ),
    )
    processor = object.__new__(Qwen3ASRMultiModalProcessor)
    processor.info = info
    out_mm_kwargs = SimpleNamespace(
        get_data=lambda: {"audio_embeds": embeddings}
    )

    updates = processor._get_prompt_updates(mm_items, {}, out_mm_kwargs)

    assert callable(updates[0].replacement)
    assert updates[0].replacement(0) == [7, 7, 7]


def test_qwen3_asr_field_config_batches_audio_embeddings():
    assert "audio_embeds" in _qwen3asr_field_config({})
