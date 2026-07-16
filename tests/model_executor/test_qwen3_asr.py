# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def test_audio_length_metadata_stays_on_cpu():
    from vllm.model_executor.models.qwen3_asr import _qwen3asr_field_config

    config = _qwen3asr_field_config(
        {"audio_feature_lengths": torch.tensor([100, 200])}
    )

    assert config["audio_feature_lengths"].field.keep_on_cpu
    assert config["feature_attention_mask"].field.keep_on_cpu
