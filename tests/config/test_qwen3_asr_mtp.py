# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.qwen3_asr_mtp import (
    Qwen3ASRMultiTokenPredictor,
    remap_qwen3_asr_mtp_weight_name,
)


def _qwen3_asr_config(depth: int | None) -> PretrainedConfig:
    text_config = PretrainedConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    text_config.model_type = "qwen3"
    thinker_config = PretrainedConfig(text_config=text_config)
    config = PretrainedConfig(
        architectures=["Qwen3ASRForConditionalGeneration"],
        thinker_config=thinker_config,
    )
    config.model_type = "qwen3_asr"
    if depth is not None:
        config.mtp_num_hidden_layers = depth
    return config


def test_qwen3_asr_mtp_draft_config_promotes_text_config():
    draft = SpeculativeConfig.hf_config_override(_qwen3_asr_config(3))
    assert draft.model_type == "qwen3_asr_mtp"
    assert draft.architectures == ["Qwen3ASRMTP"]
    assert draft.n_predict == 3
    assert draft.mtp_num_hidden_layers == 3


def test_qwen3_asr_mtp_config_requires_trained_depth():
    with pytest.raises(ValueError, match="mtp_num_hidden_layers"):
        SpeculativeConfig.hf_config_override(_qwen3_asr_config(None))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("mtp.layers.2.projection.weight", "model.layers.2.projection.weight"),
        (
            "thinker.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        ("thinker.model.norm.weight", "model.norm.weight"),
        ("thinker.lm_head.weight", "lm_head.weight"),
        ("thinker.audio_tower.proj1.weight", None),
    ],
)
def test_qwen3_asr_mtp_weight_remapping(source: str, expected: str | None):
    assert remap_qwen3_asr_mtp_weight_name(source) == expected


class _FakeMTPBlock(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, positions, previous_hidden_states, inputs_embeds):
        return torch.full_like(previous_hidden_states, self.value)


def test_qwen3_asr_mtp_selects_serial_branch_by_spec_step():
    predictor = Qwen3ASRMultiTokenPredictor.__new__(Qwen3ASRMultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.num_mtp_layers = 3
    predictor.embed_tokens = nn.Embedding(8, 4)
    predictor.layers = nn.ModuleList(
        [_FakeMTPBlock(1.0), _FakeMTPBlock(2.0), _FakeMTPBlock(3.0)]
    )
    predictor.norm = nn.Identity()
    input_ids = torch.tensor([1])
    hidden_states = torch.zeros(1, 4)

    output = predictor(
        input_ids,
        torch.tensor([0]),
        hidden_states,
        spec_step_idx=2,
    )

    assert torch.equal(output, torch.full_like(hidden_states, 3.0))
