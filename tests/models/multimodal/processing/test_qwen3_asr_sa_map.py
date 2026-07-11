# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.models.qwen3_asr import (
    _get_sa_map_output_lengths,
    _sa_map_compress,
)
from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRConfig
from vllm.transformers_utils.processors.qwen3_asr import (
    _get_sa_map_output_lengths as get_processor_sa_map_output_lengths,
)


def test_sa_map_output_lengths_match_processor():
    input_lengths = torch.tensor([100, 400, 1000])

    model_lengths = _get_sa_map_output_lengths(input_lengths, 0.6)
    processor_lengths = get_processor_sa_map_output_lengths(input_lengths, 0.6)
    numpy_lengths = get_processor_sa_map_output_lengths(
        input_lengths.numpy(), 0.6
    )

    torch.testing.assert_close(model_lengths, processor_lengths)
    np.testing.assert_array_equal(model_lengths.numpy(), numpy_lengths)


def test_sa_map_disabled_preserves_features():
    features = torch.randn(6, 4)
    importance = torch.rand(6)

    output = _sa_map_compress(features, importance, 6, 0.8)

    assert output.data_ptr() == features.data_ptr()


def test_sa_map_attention_weighted_merging():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    importance = torch.tensor([1.0, 3.0, 2.0, 2.0])

    output = _sa_map_compress(features, importance, 2, 0.99)

    torch.testing.assert_close(output, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_sa_map_pruning_has_fixed_length_and_finite_output():
    features = torch.eye(6)
    importance = torch.tensor([10.0, 1.0, 8.0, 1.0, 6.0, 1.0])

    output = _sa_map_compress(features, importance, 3, 1.0)

    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()
    assert output.argmax(dim=-1).tolist() == [0, 2, 4]


def test_sa_map_overmerge_is_limited_to_target_length():
    features = torch.ones(8, 3)
    importance = torch.arange(1, 9, dtype=torch.float32)

    output = _sa_map_compress(features, importance, 5, 0.5)

    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sa_map_retention_ratio", 0.0),
        ("sa_map_retention_ratio", 1.1),
        ("sa_map_similarity_threshold", -1.1),
        ("sa_map_similarity_threshold", 1.1),
    ],
)
def test_sa_map_config_validation(field: str, value: float):
    with pytest.raises(ValueError):
        Qwen3ASRConfig(**{field: value})
