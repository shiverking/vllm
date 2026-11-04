# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.fixed_prefix import FixedPrefixRegistry
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def test_pinned_idle_block_survives_pressure_and_releases():
    pool = BlockPool(8, True, 16)
    pinned = pool.get_new_blocks(1)
    pool.pin_blocks(pinned)
    pool.free_blocks(pinned)
    assert pinned[0].ref_cnt == 0
    assert pinned[0].pin_count == 1
    assert pool.get_num_free_blocks() == 6
    others = pool.get_new_blocks(6)
    assert pinned[0] not in others
    assert not pool.reset_prefix_cache()
    assert SingleTypeKVCacheManager._get_num_evictable_blocks(pinned) == 0
    pool.touch(pinned)
    pool.free_blocks(pinned)
    pool.free_blocks(others)
    pool.unpin_blocks(pinned)
    assert pool.get_num_free_blocks() == 7
    assert pool.reset_prefix_cache()


def test_nested_pins_and_active_release():
    pool = BlockPool(5, True, 16)
    blocks = pool.get_new_blocks(1)
    pool.pin_blocks(blocks)
    pool.pin_blocks(blocks)
    pool.unpin_blocks(blocks)
    pool.unpin_blocks(blocks)
    assert pool.get_num_free_blocks() == 3
    pool.free_blocks(blocks)
    assert pool.get_num_free_blocks() == 4


def test_registry_rejects_missing_and_mismatched_prefix():
    pool = BlockPool(100, True, 16)
    registry = FixedPrefixRegistry(SimpleNamespace(block_pool=pool))
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args={"fixed_prefix_id": "p"}),
        prompt_token_ids=[1, 3, 4],
        cache_salt="s",
        num_tokens=3,
        mm_features=[],
    )
    with pytest.raises(ValueError, match="Unknown"):
        registry.lookup(request)
    registry.entries["p"] = {"state": "ready", "tokens": [1, 2], "salt": "s"}
    with pytest.raises(ValueError, match="mismatch"):
        registry.lookup(request)


def test_partial_hybrid_capture_and_private_continuation():
    pool = BlockPool(201, True, 4)
    attention = SimpleNamespace(
        block_size=4,
        kv_cache_spec=FullAttentionSpec(
            4, num_kv_heads=1, head_size=2, dtype=torch.float32
        ),
        req_to_blocks={"warm": pool.get_new_blocks(2)},
        num_cached_block={},
    )
    recurrent = SimpleNamespace(
        block_size=4,
        kv_cache_spec=MambaSpec(
            4, shapes=((2,),), dtypes=(torch.float32,), mamba_cache_mode="align"
        ),
        req_to_blocks={"warm": pool.get_new_blocks(2)},
        num_cached_block={},
    )
    manager = SimpleNamespace(
        block_pool=pool,
        max_model_len=100,
        coordinator=SimpleNamespace(single_type_managers=[attention, recurrent]),
    )
    registry = FixedPrefixRegistry(manager)
    tokens = [1, 2, 3, 4, 5, 6]
    registry.prepare("p", tokens, "s")
    with pytest.raises(ValueError, match="budget"):
        registry.prepare("too-many", tokens, "s")
    assert "too-many" not in registry.entries
    registry.capture(
        SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args={"fixed_prefix_capture": "p"}),
            prompt_token_ids=tokens,
            max_tokens=1,
            num_computed_tokens=6,
            num_output_tokens=1,
            request_id="warm",
        )
    )
    entry = registry.entries["p"]
    assert entry["state"] == "ready"
    for m in (attention, recurrent):
        pool.free_blocks(m.req_to_blocks.pop("warm"))
    request = SimpleNamespace(request_id="image")
    for m, blocks in zip((attention, recurrent), entry["blocks"]):
        m.req_to_blocks["image"] = list(blocks)
        pool.touch(blocks)
        m.num_cached_block["image"] = 2
    assert registry.copy_count(entry) == 2
    registry.privatize(request, entry)
    assert attention.req_to_blocks["image"][0] is entry["blocks"][0][0]
    assert attention.req_to_blocks["image"][1] is not entry["blocks"][0][1]
    assert recurrent.req_to_blocks["image"][1] is not entry["blocks"][1][1]
    assert attention.num_cached_block["image"] == 1
    assert len(registry.take_copies()) == 2
    assert registry.take_copies() == []
    with pytest.raises(ValueError, match="active"):
        registry.unregister("p")
    for m in (attention, recurrent):
        pool.free_blocks(m.req_to_blocks.pop("image"))
    registry.release("image")
    registry.unregister("p")
    assert pool.get_num_free_blocks() == 200
