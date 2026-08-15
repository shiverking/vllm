# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)

pytestmark = pytest.mark.skip_global_cleanup


def _new_qwen3_asr_model() -> Qwen3ASRForConditionalGeneration:
    model = object.__new__(Qwen3ASRForConditionalGeneration)
    torch.nn.Module.__init__(model)
    return model


def test_qwen3_asr_supports_eagle3_aux_hidden_states() -> None:
    model = _new_qwen3_asr_model()
    core_model = object.__new__(Qwen3Model)
    torch.nn.Module.__init__(core_model)

    language_model = torch.nn.Module()
    language_model.model = core_model
    language_model.embed_input_ids = MagicMock()
    model.language_model = language_model

    assert supports_eagle3(model)

    aux_hidden_state_layers = (3, 15, 26)
    model.set_aux_hidden_state_layers(aux_hidden_state_layers)

    assert core_model.aux_hidden_state_layers == aux_hidden_state_layers


def test_qwen3_asr_forward_still_delegates_to_language_model() -> None:
    model = _new_qwen3_asr_model()
    expected = object()
    language_model = torch.nn.Module()
    language_model.model = MagicMock(return_value=expected)
    model.language_model = language_model

    result = model.forward(
        input_ids=None,
        positions=MagicMock(),
        intermediate_tensors=None,
        inputs_embeds=MagicMock(),
    )

    assert result is expected
    language_model.model.assert_called_once()
