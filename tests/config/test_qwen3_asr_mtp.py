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


def _qwen3_asr_config(
    depth: int | None, position_mode: str = "base"
) -> PretrainedConfig:
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
    config.mtp_branch_position_mode = position_mode
    return config


def test_qwen3_asr_mtp_draft_config_promotes_text_config():
    draft = SpeculativeConfig.hf_config_override(_qwen3_asr_config(3))
    assert draft.model_type == "qwen3_asr_mtp"
    assert draft.architectures == ["Qwen3ASRMTP"]
    assert draft.n_predict == 3
    assert draft.mtp_num_hidden_layers == 3
    assert draft.mtp_branch_position_mode == "base"


def test_qwen3_asr_mtp5_draft_config_preserves_depth_and_position_mode():
    draft = SpeculativeConfig.hf_config_override(
        _qwen3_asr_config(5, position_mode="shifted")
    )
    assert draft.n_predict == 5
    assert draft.mtp_num_hidden_layers == 5
    assert draft.mtp_branch_position_mode == "shifted"


def test_qwen3_asr_mtp_config_rejects_unknown_position_mode():
    with pytest.raises(ValueError, match="mtp_branch_position_mode"):
        SpeculativeConfig.hf_config_override(
            _qwen3_asr_config(5, position_mode="unknown")
        )


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


@pytest.mark.parametrize("spec_step_idx", range(5))
def test_qwen3_asr_mtp5_selects_serial_branch_by_spec_step(spec_step_idx: int):
    predictor = Qwen3ASRMultiTokenPredictor.__new__(Qwen3ASRMultiTokenPredictor)
    nn.Module.__init__(predictor)
    predictor.num_mtp_layers = 5
    predictor.embed_tokens = nn.Embedding(8, 4)
    predictor.layers = nn.ModuleList(
        [_FakeMTPBlock(float(index)) for index in range(1, 6)]
    )
    predictor.norm = nn.Identity()
    input_ids = torch.tensor([1])
    hidden_states = torch.zeros(1, 4)

    output = predictor(
        input_ids,
        torch.tensor([0]),
        hidden_states,
        spec_step_idx=spec_step_idx,
    )

    assert torch.equal(
        output, torch.full_like(hidden_states, float(spec_step_idx + 1))
    )
