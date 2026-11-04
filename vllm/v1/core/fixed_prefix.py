# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-engine, bounded immutable hybrid prefix checkpoints.

Registration is explicit. Unlike APC, the final attention page may be partial;
that page and every recurrent state are copied before continuation.
"""

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


class FixedPrefixRegistry:
    def __init__(self, manager, budget_fraction=0.02):
        self.manager = manager
        self.pool = manager.block_pool
        self.budget = int((self.pool.num_gpu_blocks - 1) * budget_fraction)
        self.entries = {}
        self.pending_copies = []
        self.hits = 0
        self.reused_tokens = 0
        self.copy_operations = 0

    def prepare(self, prefix_id, token_ids, cache_salt):
        if prefix_id in self.entries:
            raise ValueError("Prefix already registered")
        if not token_ids or len(token_ids) >= self.manager.max_model_len:
            raise ValueError("Invalid prefix length")
        managers = self.manager.coordinator.single_type_managers
        sizes = {m.block_size for m in managers}
        if len(sizes) != 1:
            raise ValueError("Fixed prefix requires equal hybrid block sizes")
        size = sizes.pop()
        required = 0
        for m in managers:
            spec = m.kv_cache_spec
            if isinstance(spec, MambaSpec):
                if spec.mamba_cache_mode != "align" or spec.num_speculative_blocks:
                    raise ValueError("Requires align mode without speculation")
                required += 1
            elif type(spec) is FullAttentionSpec:
                required += (len(token_ids) + size - 1) // size
            else:
                raise ValueError(f"Unsupported cache specification: {type(spec)}")
        if (
            required + sum(e["reserved_blocks"] for e in self.entries.values())
            > self.budget
        ):
            raise ValueError("Fixed prefix budget exceeded")
        self.entries[prefix_id] = {
            "tokens": list(token_ids),
            "salt": cache_salt,
            "blocks": None,
            "reserved_blocks": required,
            "block_size": size,
            "state": "warming",
            "users": set(),
        }
        return self.status(prefix_id)

    def capture(self, request):
        extra = request.sampling_params.extra_args if request.sampling_params else None
        prefix_id = (extra or {}).get("fixed_prefix_capture")
        if not prefix_id:
            return
        entry = self.entries.get(prefix_id)
        if entry is None or entry["state"] != "warming":
            return
        n = len(entry["tokens"])
        if (
            request.prompt_token_ids != entry["tokens"]
            or request.max_tokens != 1
            or request.num_computed_tokens < n
            or request.num_output_tokens != 1
        ):
            entry["state"] = "failed"
            entry["failure"] = {
                "prompt_matches": request.prompt_token_ids == entry["tokens"],
                "computed": request.num_computed_tokens,
                "expected": n,
                "outputs": request.num_output_tokens,
            }
            return
        groups = []
        for m in self.manager.coordinator.single_type_managers:
            blocks = list(m.req_to_blocks[request.request_id])
            if isinstance(m.kv_cache_spec, MambaSpec):
                # Earlier chunk checkpoints can still be owned by the warmup
                # request until its next scheduling step. Retain only the state
                # at the exact prefix boundary, not those obsolete checkpoints.
                last = (n - 1) // m.block_size
                blocks = [self.pool.null_block] * last + [blocks[last]]
            groups.append(blocks)
        unique = {b.block_id: b for group in groups for b in group if not b.is_null}
        if len(unique) > entry["reserved_blocks"]:
            entry["state"] = "failed"
            entry["failure"] = {
                "blocks": len(unique),
                "reserved": entry["reserved_blocks"],
            }
            return
        self.pool.pin_blocks(list(unique.values()))
        entry["blocks"] = tuple(groups)
        entry["state"] = "ready"

    def lookup(self, request):
        extra = request.sampling_params.extra_args if request.sampling_params else None
        prefix_id = (extra or {}).get("fixed_prefix_id")
        if not prefix_id:
            return None
        entry = self.entries.get(prefix_id)
        if entry is None or entry["state"] != "ready":
            raise ValueError("Unknown or unavailable fixed prefix")
        n = len(entry["tokens"])
        if (
            request.prompt_token_ids[:n] != entry["tokens"]
            or request.cache_salt != entry["salt"]
            or request.num_tokens <= n
        ):
            raise ValueError("Fixed prefix tokens or cache namespace mismatch")
        if any(f.mm_position.offset < n for f in request.mm_features):
            raise ValueError("Fixed prefix cannot contain image-dependent tokens")
        return entry

    def copy_count(self, entry):
        return sum(
            isinstance(m.kv_cache_spec, MambaSpec)
            or len(entry["tokens"]) % m.block_size != 0
            for m in self.manager.coordinator.single_type_managers
        )

    def privatize(self, request, entry):
        entry["users"].add(request.request_id)
        n = len(entry["tokens"])
        for group_id, m in enumerate(self.manager.coordinator.single_type_managers):
            recurrent = isinstance(m.kv_cache_spec, MambaSpec)
            if not recurrent and n % m.block_size == 0:
                continue
            blocks = m.req_to_blocks[request.request_id]
            index = (n - 1) // m.block_size
            source = blocks[index]
            target = self.pool.get_new_blocks(1)[0]
            blocks[index] = target
            self.pool.free_blocks([source])
            if not recurrent:
                m.num_cached_block[request.request_id] = n // m.block_size
            self.pending_copies.append((group_id, source.block_id, target.block_id))
            self.copy_operations += 1
        self.hits += 1
        self.reused_tokens += n

    def take_copies(self):
        copies, self.pending_copies = self.pending_copies, []
        return copies

    def release(self, request_id):
        for entry in self.entries.values():
            entry["users"].discard(request_id)

    def status(self, prefix_id):
        entry = self.entries[prefix_id]
        return {
            "prefix_id": prefix_id,
            "state": entry["state"],
            "prefix_tokens": len(entry["tokens"]),
            "reserved_blocks": entry["reserved_blocks"],
            "budget_blocks": self.budget,
            "hits": self.hits,
            "reused_tokens": self.reused_tokens,
            "copy_operations": self.copy_operations,
            "active_requests": len(entry["users"]),
            "failure": entry.get("failure"),
            "retained_blocks": [
                {"block_id": b.block_id, "pin_count": b.pin_count}
                for group in (entry["blocks"] or ())
                for b in group
                if not b.is_null
            ],
        }

    def unregister(self, prefix_id):
        entry = self.entries[prefix_id]
        if entry["users"]:
            raise ValueError("Prefix has active requests; retry unregister later")
        if entry["blocks"] is not None:
            unique = {
                b.block_id: b
                for group in entry["blocks"]
                for b in group
                if not b.is_null
            }
            if any(b.ref_cnt for b in unique.values()):
                raise ValueError("Prefix has active requests; retry unregister later")
            self.pool.unpin_blocks(list(unique.values()))
        del self.entries[prefix_id]
        return {"prefix_id": prefix_id, "state": "removed"}
