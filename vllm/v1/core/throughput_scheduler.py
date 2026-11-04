# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional logical output-growth reservations; never allocates cache pages."""

import math


class ThroughputAdmission:
    def __init__(self, config):
        defaults = dict(
            normal_output_tokens=512,
            recovery_output_tokens=1024,
            growth_step_tokens=256,
            min_remaining_tokens=128,
            safety_fraction=0.02,
            queue_scan_limit=8,
            aging_seconds=30,
        )
        unknown = set(config) - set(defaults) - {"enabled"}
        if unknown:
            raise ValueError(f"Unknown throughput scheduler settings: {unknown}")
        defaults.update({k: v for k, v in config.items() if k != "enabled"})
        for key, value in defaults.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid throughput scheduler setting: {key}")
            if key not in ("safety_fraction", "aging_seconds") and not isinstance(
                value, int
            ):
                raise ValueError(f"Expected integer: {key}")
        if defaults["safety_fraction"] >= 1:
            raise ValueError("safety_fraction must be less than one")
        self.options = defaults
        self.blocked = 0
        self.last_growth = 0

    def target(self, request, max_model_len):
        o = self.options
        generated = request.num_tokens - request.num_prompt_tokens
        initial = (
            o["recovery_output_tokens"]
            if request.max_tokens > 4096
            else o["normal_output_tokens"]
        )
        step = o["growth_step_tokens"]
        output = max(
            initial, math.ceil((generated + o["min_remaining_tokens"]) / step) * step
        )
        return min(
            max_model_len, request.num_prompt_tokens + min(request.max_tokens, output)
        )

    def needed(self, manager, request, blocks, computed):
        target = self.target(request, manager.max_model_len)
        # Some managers use defaultdict lookups even during estimation.
        absent = [
            m
            for m in getattr(manager.coordinator, "single_type_managers", [])
            if request.request_id not in m.req_to_blocks
        ]
        try:
            return manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=target,
                new_computed_blocks=blocks,
                num_encoder_tokens=0,
                total_computed_tokens=computed,
                num_tokens_main_model=target,
                apply_admission_cap=True,
            )
        finally:
            for m in absent:
                m.req_to_blocks.pop(request.request_id, None)

    def allows(self, manager, running, request, blocks, computed):
        empty = manager.empty_kv_cache_blocks.blocks
        growth = sum(
            self.needed(manager, r, empty, r.num_computed_tokens) for r in running
        )
        self.last_growth = growth
        need = self.needed(manager, request, blocks, computed)
        # Fixed-prefix continuation copies its partial attention tail and
        # recurrent checkpoints into private pages. Account conservatively.
        registry = manager.fixed_prefix_registry
        extra = request.sampling_params.extra_args or {}
        prefix_id = extra.get("fixed_prefix_id")
        if registry and prefix_id:
            need += registry.copy_count(registry.entries[prefix_id])
        safety = math.ceil(
            manager.block_pool.num_gpu_blocks * self.options["safety_fraction"]
        )
        allowed = growth + need + safety <= manager.block_pool.get_num_free_blocks()
        # An individually oversized request must be able to drain alone.
        if not running:
            allowed = True
        if not allowed:
            self.blocked += 1
        return allowed
