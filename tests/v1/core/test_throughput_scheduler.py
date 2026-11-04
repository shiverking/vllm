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
