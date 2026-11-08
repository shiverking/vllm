# Ascend LeapQuant experiment

The `cloud-drive-leapquant-ascend-exp` branch pairs generic vLLM cache and
admission changes with an optional Ascend GDN state-compression implementation
in the matching `shiverking/vllm-ascend` branch. No application repository or
image dataset is needed to build or use the implementation.

The AscendC kernel, build switches, Python binding, GDN prefill/Decode paths,
and synthetic tests belong to vLLM-Ascend. See
`csrc/experimental/leapquant/README.md` in that checkout for build, serving,
verification, and benchmark commands. Kernel arithmetic is not duplicated in
the platform-independent vLLM model or scheduler.

## Independent switches

All features are off unless explicitly configured in `--additional-config`:

```json
{
  "fixed_prefix_cache": {"budget_fraction": 0.02},
  "throughput_scheduler": {
    "enabled": true,
    "reservation_mode": "aggressive",
    "near_tokens": 1,
    "safety_fraction": 0.005
  },
  "leapquant_config": {
    "enabled": true,
    "library": "/absolute/path/to/libleapquant_native.so",
    "cores": 40
  }
}
```

`leapquant_config` is validated and consumed by the Ascend plugin. It compresses
the GDN recurrent state, retaining BF16 model weights and full-attention KV.
It does not enable attention KV quantization. The tested model is Qwen3.5-9B,
with TP/PP/PCP/DCP=1; speculative decoding and cache transfer are unsupported
by this experimental compressed-state path.

Fixed-prefix retention and throughput admission already live in this vLLM
branch. Their operation does not depend on the application client's code.
Registration of a fixed prefix captures the text before the first image,
including the hybrid model's recurrent-state checkpoint. Prefix cache entries
are immutable and bounded by the configured budget.

## Request and measurement behavior

Register the usual one-image chat request at `POST /v1/fixed-prefixes`; use the
returned prefix ID with `POST /v1/fixed-prefix/completions`. Query or delete
the registration through `/v1/fixed-prefixes/{prefix_id}`. The registration
requires a real image request for template rendering; subsequent image identity
must be unique when measuring independent photos. See the endpoint payload
fields in `vllm/entrypoints/openai/chat_completion/fixed_prefix.py`.

`max_num_seqs` is the server ceiling. It does not ensure all requested client
concurrency fits in the available cache. The aggressive policy reserves only
near-term growth and a small shared safety margin, so preemption remains
possible. Observe `/metrics` and real active batches instead of assuming that
client concurrency equals effective Decode batch size.

Compare configurations with the same model, prompt, output limits, cache
identity policy, graph sizes, and fixed observation interval. Replenish requests
throughout the interval and report drain separately. Comparing a short
Decode-only tail with an interval that includes ongoing prefill gives a biased
throughput conclusion. A larger compressed-state capacity alone does not prove
that the compressed operator or end-to-end workload runs faster.
