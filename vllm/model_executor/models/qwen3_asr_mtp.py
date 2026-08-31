# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only ParaASR-style MTP model for Qwen3-ASR."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors


def _get_qwen3_asr_mtp_config(vllm_config: VllmConfig) -> Any:
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or speculative_config.draft_model_config is None
    ):
        raise ValueError("Qwen3-ASR MTP requires a draft model config")
    return speculative_config.draft_model_config.hf_config


def remap_qwen3_asr_mtp_weight_name(name: str) -> str | None:
    """Map the self-contained Qwen3-ASR checkpoint layout to this model."""
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("thinker.model.embed_tokens."):
        return name.replace("thinker.model.embed_tokens.", "model.embed_tokens.", 1)
    if name.startswith("thinker.model.norm."):
        return name.replace("thinker.model.norm.", "model.norm.", 1)
    if name.startswith("thinker.lm_head."):
        return name.replace("thinker.lm_head.", "lm_head.", 1)
    return None


class Qwen3ASRMTPBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = _get_qwen3_asr_mtp_config(vllm_config)
        quant_config = vllm_config.quant_config
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.projection = ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projection"),
        )
        self.decoder_layer = Qwen3DecoderLayer(
            config=config,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "decoder_layer"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        fused, _ = self.projection(
            torch.cat(
                [
                    self.hidden_norm(previous_hidden_states),
                    self.embedding_norm(inputs_embeds),
                ],
                dim=-1,
            )
        )
        hidden_states, residual = self.decoder_layer(
            positions=positions,
            hidden_states=fused,
            residual=None,
        )
        return hidden_states + residual


class Qwen3ASRMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _get_qwen3_asr_mtp_config(vllm_config)
        self.num_mtp_layers = int(config.mtp_num_hidden_layers)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            Qwen3ASRMTPBlock(vllm_config, maybe_prefix(prefix, f"layers.{index}"))
            for index in range(self.num_mtp_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        layer = self.layers[spec_step_idx % self.num_mtp_layers]
        return self.norm(layer(positions, previous_hidden_states, inputs_embeds))


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3ASRMTP(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = _get_qwen3_asr_mtp_config(vllm_config)
        self.model = Qwen3ASRMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            if self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise NotImplementedError("Qwen3-ASR MTP currently supports PP=1 only")
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for checkpoint_name, checkpoint_weight in weights:
            name = remap_qwen3_asr_mtp_weight_name(checkpoint_name)
            if name is None or "rotary_emb.inv_freq" in name:
                continue
            for packed_name, source_name, shard_id in stacked_params_mapping:
                if source_name not in name:
                    continue
                mapped_name = name.replace(source_name, packed_name)
                if mapped_name not in params:
                    continue
                param = params[mapped_name]
                param.weight_loader(param, checkpoint_weight, shard_id)
                loaded.add(mapped_name)
                break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected Qwen3-ASR MTP weight: {checkpoint_name}")
                param = params[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, checkpoint_weight)
                loaded.add(name)
        return loaded
