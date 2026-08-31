# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.qwen3_asr_mtp import (
    Qwen3ASRMultiTokenPredictor,
    remap_qwen3_asr_mtp_weight_name,
)
from vllm.transformers_utils.config import get_config


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


def test_qwen3_asr_mtp_override_accepts_parser_text_config_shape():
    config = PretrainedConfig(
        architectures=["Qwen3ASRForConditionalGeneration"],
        mtp_num_hidden_layers=5,
        mtp_branch_position_mode="base",
    )
    config.model_type = "qwen3_asr_text"

    draft = SpeculativeConfig.hf_config_override(config)

    assert draft.model_type == "qwen3_asr_mtp"
    assert draft.architectures == ["Qwen3ASRMTP"]
    assert draft.n_predict == 5


def test_qwen3_asr_mtp_config_rejects_unknown_position_mode():
    with pytest.raises(ValueError, match="mtp_branch_position_mode"):
        SpeculativeConfig.hf_config_override(
            _qwen3_asr_config(5, position_mode="unknown")
        )


def test_qwen3_asr_mtp_override_through_hf_config_parser(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "model_type": "qwen3_asr",
                "mtp_num_hidden_layers": 5,
                "num_nextn_predict_layers": 5,
                "mtp_branch_position_mode": "base",
                "thinker_config": {
                    "audio_config": {
                        "model_type": "qwen3_asr_audio_encoder",
                    },
                    "text_config": {
                        "model_type": "qwen3",
                        "vocab_size": 32,
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    draft = get_config(
        tmp_path,
        trust_remote_code=False,
        hf_overrides_fn=SpeculativeConfig.hf_config_override,
    )

    assert draft.model_type == "qwen3_asr_mtp"
    assert draft.architectures == ["Qwen3ASRMTP"]
    assert draft.mtp_num_hidden_layers == 5
    assert draft.mtp_branch_position_mode == "base"


def test_qwen3_asr_spec_config_repairs_composite_draft_model_config():
    target_hf_config = _qwen3_asr_config(5)
    target_model_config = SimpleNamespace(
        architectures=["Qwen3ASRForConditionalGeneration"],
        hf_config=target_hf_config,
        hf_text_config=target_hf_config.thinker_config.text_config,
        model="checkpoint",
        tokenizer="checkpoint",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype=torch.bfloat16,
        seed=0,
        revision=None,
        code_revision=None,
        tokenizer_revision=None,
        max_model_len=8192,
        quantization=None,
        enforce_eager=True,
        max_logprobs=20,
        config_format="auto",
    )
    target_parallel_config = SimpleNamespace(tensor_parallel_size=1)
    incorrectly_parsed_draft = SimpleNamespace(
        model="checkpoint",
        hf_config=target_hf_config,
        max_model_len=65536,
        encoder_config=object(),
        hf_image_processor_config=object(),
        multimodal_config=object(),
        attention_chunk_size=None,
    )

    def refresh_architecture(spec_config):
        spec_config.draft_model_config.hf_text_config = (
            spec_config.draft_model_config.hf_config
        )
        spec_config.draft_model_config.model_arch_config = SimpleNamespace(
            architectures=spec_config.draft_model_config.hf_config.architectures
        )

    with (
        patch(
            "vllm.config.speculative.ModelConfig",
            return_value=incorrectly_parsed_draft,
        ),
        patch.object(
            SpeculativeConfig,
            "update_arch_",
            refresh_architecture,
        ),
        patch.object(
            SpeculativeConfig,
            "_verify_and_get_draft_tp",
            return_value=1,
        ),
        patch.object(
            SpeculativeConfig,
            "create_draft_parallel_config",
            return_value=target_parallel_config,
        ),
        patch.object(
            SpeculativeConfig,
            "_maybe_override_draft_max_model_len",
            return_value=8192,
        ),
    ):
        config = SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=5,
            target_model_config=target_model_config,
            target_parallel_config=target_parallel_config,
        )

    assert config.draft_model_config.hf_config.model_type == "qwen3_asr_mtp"
    assert config.draft_model_config.hf_config.architectures == ["Qwen3ASRMTP"]
    assert config.draft_model_config.encoder_config is None
    assert config.draft_model_config.multimodal_config is None


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
