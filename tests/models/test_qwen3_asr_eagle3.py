# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
)


def test_qwen3_asr_supports_eagle3_aux_hidden_states() -> None:
    model = object.__new__(Qwen3ASRForConditionalGeneration)
    language_model = object.__new__(Qwen3Model)
    language_model.__dict__["aux_hidden_state_layers"] = ()
    model.__dict__["language_model"] = SimpleNamespace(model=language_model)

    assert supports_eagle3(model)

    aux_hidden_state_layers = (3, 15, 26)
    model.set_aux_hidden_state_layers(aux_hidden_state_layers)

    assert language_model.aux_hidden_state_layers == aux_hidden_state_layers


def test_qwen3_asr_forward_still_delegates_to_language_model() -> None:
    model = object.__new__(Qwen3ASRForConditionalGeneration)
    expected = object()
    language_model = MagicMock(return_value=expected)
    model.__dict__["language_model"] = SimpleNamespace(model=language_model)

    result = model.forward(
        input_ids=None,
        positions=MagicMock(),
        intermediate_tensors=None,
        inputs_embeds=MagicMock(),
    )

    assert result is expected
    language_model.assert_called_once()
