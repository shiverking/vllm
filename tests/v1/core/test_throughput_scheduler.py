# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS

import pytest

from vllm.v1.core.throughput_scheduler import ThroughputAdmission

pytestmark = pytest.mark.skip_global_cleanup


def req(generated=0, maximum=4096):
    return NS(
        num_tokens=4062 + generated,
        num_prompt_tokens=4062,
        max_tokens=maximum,
        request_id="a",
        num_computed_tokens=2024,
        sampling_params=NS(extra_args={}),
    )


def test_output_growth_and_caps():
    a = ThroughputAdmission({})
    assert a.target(req(), 16384) == 4574
    assert a.target(req(500), 16384) == 4830
    assert a.target(req(maximum=8192), 16384) == 5086
    assert a.target(req(4090), 16384) == 8158
    assert a.target(req(), 4200) == 4200


def test_admission_accounts_for_running_growth():
    a = ThroughputAdmission({})
    manager = NS(
        max_model_len=16384,
        empty_kv_cache_blocks=NS(blocks=((),)),
        coordinator=NS(get_num_blocks_to_allocate=lambda **kw: 10),
        block_pool=NS(num_gpu_blocks=100, get_num_free_blocks=lambda: 21),
        fixed_prefix_registry=None,
    )
    assert not a.allows(manager, [req()], req(), ((),), 2024)
    manager.block_pool.get_num_free_blocks = lambda: 22
    assert a.allows(manager, [req()], req(), ((),), 2024)
    assert a.allows(manager, [], req(), ((),), 2024)


@pytest.mark.parametrize(
    "config",
    [
        {"safety_fraction": 1},
        {"queue_scan_limit": 1.5},
        {"unknown": 1},
        {"growth_step_tokens": 0},
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        ThroughputAdmission(config)


def test_soft_reservation_rounding_and_lifecycle():
    a = ThroughputAdmission({"reservation_mode": "soft"})
    m = NS(
        max_model_len=16384,
        empty_kv_cache_blocks=NS(blocks=((),)),
        coordinator=NS(get_num_blocks_to_allocate=lambda **kw: 10),
        block_pool=NS(num_gpu_blocks=100, get_num_free_blocks=lambda: 14),
        fixed_prefix_registry=None,
    )
    assert not a.allows(m, [req()], req(), ((),), 2024)
    assert a.telemetry["last_discounted_growth_blocks"] == 3
    m.block_pool.get_num_free_blocks = lambda: 15
    assert a.allows(m, [req()], req(), ((),), 2024)
    assert not a.blocked_since
    m.block_pool.get_num_free_blocks = lambda: 0
    assert not a.allows(m, [req()], req(), ((),), 2024)
    a.begin_step({})
    assert not a.blocked_since


@pytest.mark.parametrize(
    "config",
    [
        {"reservation_mode": "bad"},
        {"growth_reservation_fraction": 0},
        {"growth_reservation_fraction": 1.1},
    ],
)
def test_invalid_soft_settings(config):
    with pytest.raises(ValueError):
        ThroughputAdmission(config)


def test_wait_age_uses_arrival_wall_clock(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.core.throughput_scheduler.time.time", lambda: 1700000040
    )
    assert ThroughputAdmission.wait_age(NS(arrival_time=1700000000)) == 40
    assert ThroughputAdmission.wait_age(NS(arrival_time=1700000050)) == 0


def test_boundary_targets_and_shared_cap():
    a = ThroughputAdmission({"reservation_mode": "boundary"})
    assert a.near_target(req(), 16384) == 4078
    assert a.near_target(req(4090), 16384) == 8158
    assert a.near_target(req(), 4070) == 4070
    # Already owned pages are excluded by the coordinator; only future
    # increments are counted. Near=2, far delta=98: shared cap is 5 blocks.
    m = NS(
        max_model_len=16384,
        empty_kv_cache_blocks=NS(blocks=((),)),
        coordinator=NS(
            get_num_blocks_to_allocate=lambda **kw: (
                2 if kw["num_tokens"] < 4100 else 100
            )
        ),
        block_pool=NS(num_gpu_blocks=100, get_num_free_blocks=lambda: 10),
        fixed_prefix_registry=None,
    )
    assert not a.allows(m, [req()], req(), ((),), 2024)
    assert a.telemetry["last_near_growth_blocks"] == 2
    assert a.telemetry["last_far_growth_blocks"] == 98
    assert a.telemetry["last_discounted_growth_blocks"] == 7
    m.block_pool.get_num_free_blocks = lambda: 11
    assert a.allows(m, [req()], req(), ((),), 2024)


@pytest.mark.parametrize(
    "config",
    [
        {"near_tokens": 0},
        {"near_tokens": 1.5},
        {"far_reservation_fraction": 1.1},
        {"far_pool_cap_fraction": 0},
    ],
)
def test_invalid_boundary_settings(config):
    with pytest.raises(ValueError):
        ThroughputAdmission(config)
