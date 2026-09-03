# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate stable-window reuse for the Qwen3-ASR audio encoder.

This is an offline diagnostic. It intentionally does not add a production
cache or require changes to the Qwen3-ASR WebSocket service.

The optional manifest is JSONL. Each row contains ``path`` and may contain
``id``, ``language``, ``request_kind``, and an exact ``prompt``. Relative audio
paths are resolved against the manifest directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_PREFIX_SECONDS = (1.92, 2.0, 2.08, 3.92, 4.0, 4.08, 7.92, 8.0, 8.08)
DEFAULT_BENCHMARK_SECONDS = (4.0, 6.0, 8.0, 12.0, 20.0, 30.0)
DEFAULT_AUDIO_GRAPH_SIZES = (26, 52, 78, 128, 256, 384, 512)
DEFAULT_DECODE_GRAPH_SIZES = tuple(range(2, 41, 2))


@dataclass(frozen=True)
class WindowGeometry:
    n_window: int
    n_window_infer: int
    conv_chunksize: int
    conv_feature_window: int
    attention_feature_window: int
    conv_chunks_per_attention_window: int
    encoder_tokens_per_attention_window: int
    sampling_rate: int
    hop_length: int
    attention_window_seconds: float


@dataclass(frozen=True)
class ModelLoadingResolution:
    quantization: str | None
    load_format: str
    modelslim_config_found: bool


@dataclass(frozen=True)
class AudioCase:
    case_id: str
    audio: np.ndarray
    language: str | None = None
    source: str = "synthetic"
    prompt: str | None = None
    request_kind: str = "final"


def resolve_model_loading(
    model_path: str,
    quantization: str,
    load_format: str,
) -> ModelLoadingResolution:
    """Resolve probe defaults without forcing ModelSlim on float models."""
    model_dir = Path(model_path)
    modelslim_config_found = (
        model_dir.is_dir()
        and (model_dir / "quant_model_description.json").is_file()
    )
    requested_quantization = quantization.lower()
    if requested_quantization == "auto":
        resolved_quantization = (
            "ascend" if modelslim_config_found else None
        )
    elif requested_quantization == "none":
        resolved_quantization = None
    else:
        if (
            requested_quantization == "ascend"
            and model_dir.is_dir()
            and not modelslim_config_found
        ):
            raise ValueError(
                "--quantization ascend requires "
                "quant_model_description.json in the local model directory; "
                "use --quantization none for float weights."
            )
        resolved_quantization = quantization

    return ModelLoadingResolution(
        quantization=resolved_quantization,
        load_format=load_format,
        modelslim_config_found=modelslim_config_found,
    )


def encoder_output_length(feature_frames: int) -> int:
    """Qwen3-ASR output-token formula for one audio item."""
    remainder = feature_frames % 100
    feature_length = (remainder - 1) // 2 + 1
    return (
        ((feature_length - 1) // 2 + 1 - 1) // 2
        + 1
        + (feature_frames // 100) * 13
    )


def derive_window_geometry(
    *,
    n_window: int,
    n_window_infer: int,
    conv_chunksize: int,
    sampling_rate: int,
    hop_length: int,
) -> WindowGeometry:
    conv_window = n_window * 2
    if n_window_infer <= 0 or conv_window <= 0:
        raise ValueError("Audio window sizes must be positive")
    if n_window_infer % conv_window:
        raise ValueError(
            "n_window_infer must be an integer multiple of n_window * 2"
        )
    return WindowGeometry(
        n_window=n_window,
        n_window_infer=n_window_infer,
        conv_chunksize=conv_chunksize,
        conv_feature_window=conv_window,
        attention_feature_window=n_window_infer,
        conv_chunks_per_attention_window=n_window_infer // conv_window,
        encoder_tokens_per_attention_window=encoder_output_length(n_window_infer),
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        attention_window_seconds=n_window_infer * hop_length / sampling_rate,
    )


def stable_feature_frames(feature_frames: int, attention_window: int) -> int:
    return feature_frames // attention_window * attention_window


def attention_sequence_lengths(
    feature_frames: int, geometry: WindowGeometry
) -> list[int]:
    """Mirror the encoder's runtime attention-topology calculation."""
    conv_window = geometry.conv_feature_window
    chunk_lengths = [conv_window] * (feature_frames // conv_window)
    if remainder := feature_frames % conv_window:
        chunk_lengths.append(remainder)
    if not chunk_lengths:
        return []

    def cnn_length(length: int) -> int:
        for _ in range(3):
            length = (length - 1) // 2 + 1
        return length

    padded_cnn_length = max(cnn_length(length) for length in chunk_lengths)
    window_aftercnn = (
        padded_cnn_length * geometry.conv_chunks_per_attention_window
    )
    total_tokens = encoder_output_length(feature_frames)
    topology = [window_aftercnn] * (total_tokens // window_aftercnn)
    if tail := total_tokens % window_aftercnn:
        topology.append(tail)
    return topology


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    shape_equal = reference.shape == candidate.shape
    if not shape_equal:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "max_abs": None,
            "mean_abs": None,
            "cosine": None,
            "allclose": False,
        }
    if reference.size == 0:
        return {
            "shape_equal": True,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "cosine": 1.0,
            "allclose": True,
        }
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    cand = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(ref - cand)
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cosine = float(np.dot(ref, cand) / denominator) if denominator else 1.0
    return {
        "shape_equal": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine": cosine,
        "allclose": bool(np.allclose(ref, cand, atol=1e-2, rtol=1e-2)),
    }


def passes_numerical_gate(
    metrics: dict[str, Any], repeat_noise: list[dict[str, Any]]
) -> bool:
    if not metrics["allclose"] or metrics["cosine"] < 0.9999:
        return False
    noise_max = max(
        (
            float(item["max_abs"])
            for item in repeat_noise
            if item.get("max_abs") is not None
        ),
        default=0.0,
    )
    if noise_max == 0.0 or metrics["max_abs"] is None:
        return True
    return float(metrics["max_abs"]) <= max(1e-2, 2.0 * noise_max)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percent))


def summarize_timings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    full = [float(row["full_encoder_ms"]) for row in rows]
    incremental = [float(row["incremental_encoder_ms"]) for row in rows]
    reductions = [float(row["compute_reduction_ratio"]) for row in rows]
    return {
        "count": len(rows),
        "full_encoder_ms": {
            "p50": percentile(full, 50),
            "p95": percentile(full, 95),
            "p99": percentile(full, 99),
        },
        "incremental_encoder_ms": {
            "p50": percentile(incremental, 50),
            "p95": percentile(incremental, 95),
            "p99": percentile(incremental, 99),
        },
        "compute_reduction_ratio": {
            "p50": percentile(reductions, 50),
            "p95": percentile(reductions, 95),
            "p99": percentile(reductions, 99),
        },
    }


def _parse_number_list(value: str, converter: type = float) -> tuple:
    return tuple(converter(item.strip()) for item in value.split(",") if item.strip())


def _read_wav(path: Path, target_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(f"Only PCM16 WAV is supported without soundfile: {path}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != target_rate:
        old_positions = np.arange(audio.size, dtype=np.float64) / rate
        new_size = int(round(audio.size * target_rate / rate))
        new_positions = np.arange(new_size, dtype=np.float64) / target_rate
        audio = np.interp(new_positions, old_positions, audio).astype(np.float32)
    return audio


def load_audio_manifest(path: Path | None, sampling_rate: int) -> list[AudioCase]:
    if path is None:
        return []
    cases: list[AudioCase] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            audio_path = Path(item["path"])
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            try:
                import soundfile as sf

                audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if rate != sampling_rate:
                    old_positions = np.arange(audio.size, dtype=np.float64) / rate
                    new_size = int(round(audio.size * sampling_rate / rate))
                    new_positions = (
                        np.arange(new_size, dtype=np.float64) / sampling_rate
                    )
                    audio = np.interp(new_positions, old_positions, audio)
                audio = np.asarray(audio, dtype=np.float32)
            except ImportError:
                audio = _read_wav(audio_path, sampling_rate)
            cases.append(
                AudioCase(
                    case_id=str(item.get("id", f"manifest-{line_number}")),
                    audio=audio,
                    language=item.get("language"),
                    source=str(audio_path),
                    prompt=item.get("prompt"),
                    request_kind=str(item.get("request_kind", "final")),
                )
            )
    return cases


def synthetic_audio_cases(
    sampling_rate: int, duration: float = 30.0
) -> list[AudioCase]:
    sample_count = int(round(duration * sampling_rate))
    positions = np.arange(sample_count, dtype=np.float32) / sampling_rate
    rng = np.random.default_rng(20260903)
    impulse = np.zeros(sample_count, dtype=np.float32)
    for second in (1.92, 2.0, 2.08, 3.92, 4.0, 4.08, 7.92, 8.0, 8.08):
        index = min(int(round(second * sampling_rate)), sample_count - 1)
        impulse[index] = 0.9
    return [
        AudioCase("synthetic-silence", np.zeros(sample_count, dtype=np.float32)),
        AudioCase("synthetic-sine", 0.2 * np.sin(2 * math.pi * 440 * positions)),
        AudioCase(
            "synthetic-noise",
            rng.normal(0, 0.05, sample_count).astype(np.float32),
        ),
        AudioCase("synthetic-boundary-impulse", impulse),
    ]


def _processor_features(processor: Any, audio: np.ndarray) -> tuple[np.ndarray, int]:
    placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
    result = processor(
        text=placeholder,
        audio=[audio],
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    features = np.asarray(result["input_features"])[0]
    mask_key = (
        "feature_attention_mask"
        if "feature_attention_mask" in result
        else "attention_mask"
    )
    feature_length = int(np.asarray(result[mask_key])[0].sum())
    return np.ascontiguousarray(features[:, :feature_length]), feature_length


def _sync_device(torch_module: Any, device_type: str) -> None:
    if device_type == "npu" and hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()
    elif device_type == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


@dataclass
class EncoderProbeCall:
    features_x: np.ndarray
    features_y: np.ndarray
    feature_length_x: int
    feature_length_y: int
    sampling_rate: int
    hop_length: int
    trace_layers: bool
    repeat_count: int = 3
    warmup_iterations: int = 0
    timing_iterations: int = 0

    def __call__(self, model: Any) -> dict[str, Any]:
        import types

        import torch

        encoder = model.audio_tower
        geometry = derive_window_geometry(
            n_window=int(encoder.n_window),
            n_window_infer=int(encoder.n_window_infer),
            conv_chunksize=int(encoder.conv_chunksize),
            sampling_rate=self.sampling_rate,
            hop_length=self.hop_length,
        )
        device = encoder.device

        def prepare(features: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(features).to(device=device, dtype=encoder.dtype)

        x = prepare(self.features_x)
        y = prepare(self.features_y)

        def output_length(length: int) -> int:
            return encoder_output_length(length)

        def run(features: torch.Tensor, length: int) -> torch.Tensor:
            feature_lens = torch.tensor([length], dtype=torch.long, device="cpu")
            aftercnn_lens = torch.tensor(
                [output_length(length)], dtype=torch.long, device="cpu"
            )
            with torch.inference_mode():
                return encoder(
                    features[:, :length],
                    feature_lens=feature_lens,
                    aftercnn_lens=aftercnn_lens,
                )

        def capture(
            features: torch.Tensor, length: int
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            traces: dict[str, torch.Tensor] = {}
            hooks = []

            def save(name: str):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(tensor, torch.Tensor):
                        traces[name] = tensor.detach().clone()

                return hook

            if self.trace_layers:
                hooks.extend(
                    [
                        encoder.conv2d1.register_forward_hook(save("conv2d1")),
                        encoder.conv2d2.register_forward_hook(save("conv2d2")),
                        encoder.conv2d3.register_forward_hook(save("conv2d3")),
                        encoder.conv_out.register_forward_hook(save("conv_out")),
                    ]
                )
                hooks.extend(
                    layer.register_forward_hook(save(f"encoder_layer_{index}"))
                    for index, layer in enumerate(encoder.layers)
                )
                original_body = encoder._forward_encoder_body

                def wrapped_body(
                    _self: Any,
                    hidden_states: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    max_seqlen: torch.Tensor | None,
                    sequence_lengths: torch.Tensor,
                ) -> torch.Tensor:
                    traces["post_position"] = hidden_states.detach().clone()
                    return original_body(
                        hidden_states, cu_seqlens, max_seqlen, sequence_lengths
                    )

                encoder._forward_encoder_body = types.MethodType(wrapped_body, encoder)
            try:
                output = run(features, length)
                traces["final_embedding"] = output.detach().clone()
                return output, traces
            finally:
                if self.trace_layers:
                    encoder._forward_encoder_body = original_body
                for hook in hooks:
                    hook.remove()

        stable_frames = stable_feature_frames(
            self.feature_length_x, geometry.attention_feature_window
        )
        stable_tokens = output_length(stable_frames) if stable_frames else 0
        stable_conv_chunks = stable_frames // geometry.conv_feature_window

        repeated = [run(x, self.feature_length_x) for _ in range(self.repeat_count)]
        output_x, traces_x = capture(x, self.feature_length_x)
        output_y, traces_y = capture(y, self.feature_length_y)

        block_outputs = []
        for start in range(0, stable_frames, geometry.attention_feature_window):
            end = start + geometry.attention_feature_window
            block_outputs.append(
                run(y[:, start:end], geometry.attention_feature_window)
            )
        tail_frames = self.feature_length_y - stable_frames
        if tail_frames:
            block_outputs.append(run(y[:, stable_frames:], tail_frames))
        incremental = (
            torch.cat(block_outputs, dim=0)
            if block_outputs
            else output_y.detach().clone()
        )

        def as_numpy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().float().cpu().numpy()

        temporal_metrics = tensor_metrics(
            as_numpy(output_x[:stable_tokens]), as_numpy(output_y[:stable_tokens])
        )
        reconstruction_metrics = tensor_metrics(
            as_numpy(output_y), as_numpy(incremental)
        )
        noise_metrics = [
            tensor_metrics(as_numpy(repeated[0]), as_numpy(item))
            for item in repeated[1:]
        ]

        trace_metrics: dict[str, Any] = {}
        for name in traces_x.keys() & traces_y.keys():
            left = traces_x[name]
            right = traces_y[name]
            if name.startswith("conv") and left.ndim >= 1:
                left = left[:stable_conv_chunks]
                right = right[:stable_conv_chunks]
            elif left.ndim >= 1:
                left = left[:stable_tokens]
                right = right[:stable_tokens]
            trace_metrics[name] = tensor_metrics(as_numpy(left), as_numpy(right))

        full_times: list[float] = []
        incremental_times: list[float] = []
        if self.timing_iterations:
            for _ in range(self.warmup_iterations):
                run(y, self.feature_length_y)
                if tail_frames:
                    run(y[:, stable_frames:], tail_frames)
            _sync_device(torch, device.type)
            for _ in range(self.timing_iterations):
                start = time.perf_counter_ns()
                run(y, self.feature_length_y)
                _sync_device(torch, device.type)
                full_times.append((time.perf_counter_ns() - start) / 1e6)

                start = time.perf_counter_ns()
                if tail_frames:
                    tail_output = run(y[:, stable_frames:], tail_frames)
                    torch.cat([*block_outputs[:-1], tail_output], dim=0)
                else:
                    torch.cat(block_outputs, dim=0)
                _sync_device(torch, device.type)
                incremental_times.append((time.perf_counter_ns() - start) / 1e6)

        full_p50 = percentile(full_times, 50)
        incremental_p50 = percentile(incremental_times, 50)
        reduction = (
            1.0 - incremental_p50 / full_p50
            if full_p50 and incremental_p50 is not None
            else None
        )
        cache_bytes = stable_tokens * output_y.shape[-1] * output_y.element_size()
        device_memory: dict[str, int] = {}
        try:
            if device.type == "npu" and hasattr(torch, "npu"):
                free_bytes, total_bytes = torch.npu.mem_get_info(device)
                device_memory = {
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "allocated_bytes": int(torch.npu.memory_allocated(device)),
                    "reserved_bytes": int(torch.npu.memory_reserved(device)),
                }
            elif device.type == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                device_memory = {
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                }
        except (AttributeError, RuntimeError, TypeError):
            device_memory = {}
        return {
            "geometry": asdict(geometry),
            "device": str(device),
            "dtype": str(encoder.dtype),
            "feature_length_x": self.feature_length_x,
            "feature_length_y": self.feature_length_y,
            "stable_feature_frames": stable_frames,
            "stable_encoder_tokens": stable_tokens,
            "tail_feature_frames": tail_frames,
            "attention_sequence_lengths_x": attention_sequence_lengths(
                self.feature_length_x, geometry
            ),
            "attention_sequence_lengths_y": attention_sequence_lengths(
                self.feature_length_y, geometry
            ),
            "cache_bytes": int(cache_bytes),
            "device_memory": device_memory,
            "temporal": temporal_metrics,
            "reconstruction": reconstruction_metrics,
            "repeat_noise": noise_metrics,
            "trace": trace_metrics,
            "full_embedding": as_numpy(output_y),
            "incremental_embedding": as_numpy(incremental),
            "timing": {
                "iterations": self.timing_iterations,
                "full_encoder_ms": {
                    "p50": full_p50,
                    "p95": percentile(full_times, 95),
                    "p99": percentile(full_times, 99),
                },
                "incremental_encoder_ms": {
                    "p50": incremental_p50,
                    "p95": percentile(incremental_times, 95),
                    "p99": percentile(incremental_times, 99),
                },
                "compute_reduction_ratio": reduction,
            },
        }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False) + "\n")


def _decoder_output(llm: Any, prompt: str, embedding: np.ndarray, max_tokens: int):
    import torch

    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
    request = {
        "prompt": prompt,
        "multi_modal_data": {"audio": torch.from_numpy(embedding)},
    }
    start = time.perf_counter_ns()
    result = llm.generate(request, params, use_tqdm=False)[0].outputs[0]
    duration_ms = (time.perf_counter_ns() - start) / 1e6
    first_logprobs = result.logprobs[0] if result.logprobs else {}
    first_token_top_ids = sorted(
        (int(token_id) for token_id in first_logprobs),
        key=lambda token_id: first_logprobs[token_id].logprob,
        reverse=True,
    )
    return {
        "token_ids": list(result.token_ids),
        "text": result.text,
        "first_token_top_id": first_token_top_ids[0]
        if first_token_top_ids
        else None,
        "first_token_logprobs": {
            str(token_id): float(logprob.logprob)
            for token_id, logprob in first_logprobs.items()
        },
        "duration_ms": duration_ms,
    }


def _raw_audio_output(
    llm: Any, prompt: str, audio: np.ndarray, max_tokens: int
) -> dict[str, Any]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
    request = {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}
    start = time.perf_counter_ns()
    result = llm.generate(request, params, use_tqdm=False)[0].outputs[0]
    return {
        "token_ids": list(result.token_ids),
        "text": result.text,
        "duration_ms": (time.perf_counter_ns() - start) / 1e6,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--append-ms", default="80,500,2000")
    parser.add_argument(
        "--prefix-seconds",
        default=",".join(str(item) for item in DEFAULT_PREFIX_SECONDS),
    )
    parser.add_argument(
        "--benchmark-seconds",
        default=",".join(str(item) for item in DEFAULT_BENCHMARK_SECONDS),
    )
    parser.add_argument(
        "--audio-encoder-aclgraph-sizes",
        default=",".join(str(item) for item in DEFAULT_AUDIO_GRAPH_SIZES),
    )
    parser.add_argument(
        "--cudagraph-capture-sizes",
        default=",".join(str(item) for item in DEFAULT_DECODE_GRAPH_SIZES),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--quantization",
        default="auto",
        help=(
            "Quantization method. 'auto' enables Ascend ModelSlim only when "
            "quant_model_description.json exists; 'none' forces float weights."
        ),
    )
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--timing-iterations", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default=(
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    )
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--case-limit", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.transformers_utils.processor import cached_processor_from_config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "run.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("qwen3-asr-incremental-encoder-probe")

    append_ms = _parse_number_list(args.append_ms, int)
    prefix_seconds = _parse_number_list(args.prefix_seconds)
    benchmark_seconds = set(_parse_number_list(args.benchmark_seconds))
    graph_sizes = _parse_number_list(args.audio_encoder_aclgraph_sizes, int)
    capture_sizes = _parse_number_list(args.cudagraph_capture_sizes, int)
    modes = ("eager", "graph") if args.mode == "both" else (args.mode,)
    model_loading = resolve_model_loading(
        args.model_path,
        args.quantization,
        args.load_format,
    )
    logger.info(
        "Resolved model loading: quantization=%s, load_format=%s, "
        "modelslim_config_found=%s",
        model_loading.quantization,
        model_loading.load_format,
        model_loading.modelslim_config_found,
    )

    config = {
        "schema_version": SCHEMA_VERSION,
        "model_path": args.model_path,
        "audio_manifest": str(args.audio_manifest) if args.audio_manifest else None,
        "mode": args.mode,
        "append_ms": append_ms,
        "prefix_seconds": prefix_seconds,
        "benchmark_seconds": sorted(benchmark_seconds),
        "audio_encoder_aclgraph_sizes": graph_sizes,
        "cudagraph_capture_sizes": capture_sizes,
        "warmup_iterations": args.warmup_iterations,
        "timing_iterations": args.timing_iterations,
        "decoder_validation": not args.skip_decoder,
        "dtype": args.dtype,
        "requested_quantization": args.quantization,
        "resolved_quantization": model_loading.quantization,
        "requested_load_format": args.load_format,
        "resolved_load_format": model_loading.load_format,
        "modelslim_config_found": model_loading.modelslim_config_found,
        "production_cache_implemented": False,
    }
    _write_json(args.output_dir / "config.json", config)
    for filename in (
        "encoder_cases.jsonl",
        "decoder_cases.jsonl",
        "timings.jsonl",
        "transcripts.csv",
    ):
        (args.output_dir / filename).write_text("", encoding="utf-8")

    encoder_rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    geometry: dict[str, Any] | None = None
    eager_embeddings: dict[tuple[str, float, float, int], np.ndarray] = {}
    eager_decoder_outputs: dict[tuple[str, float, float, int], dict[str, Any]] = {}
    cross_mode_rows: list[dict[str, Any]] = []

    for mode in modes:
        logger.info("Loading %s model mode", mode)
        llm_kwargs: dict[str, Any] = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "dtype": args.dtype,
            "enable_mm_embeds": not args.skip_decoder,
            "enforce_eager": mode == "eager",
        }
        if model_loading.quantization is not None:
            llm_kwargs["quantization"] = model_loading.quantization
        if model_loading.load_format.lower() != "auto":
            llm_kwargs["load_format"] = model_loading.load_format
        if mode == "graph":
            llm_kwargs["compilation_config"] = {
                "cudagraph_mode": "FULL",
                "cudagraph_capture_sizes": list(capture_sizes),
            }
            llm_kwargs["additional_config"] = {
                "ascend_compilation_config": {"fuse_norm_quant": False},
                "audio_encoder_aclgraph_sizes": list(graph_sizes),
            }
        llm = LLM(**llm_kwargs)
        processor = cached_processor_from_config(llm.model_config)
        sampling_rate = int(processor.feature_extractor.sampling_rate)
        hop_length = int(processor.feature_extractor.hop_length)
        cases = load_audio_manifest(args.audio_manifest, sampling_rate)
        if not args.skip_synthetic:
            cases.extend(synthetic_audio_cases(sampling_rate))
        if args.case_limit is not None:
            cases = cases[: args.case_limit]
        if not cases:
            raise ValueError("No audio cases were provided")

        max_append_seconds = max(append_ms) / 1000.0
        benchmark_prefixes = {
            duration - max_append_seconds
            for duration in benchmark_seconds
            if duration > max_append_seconds
        }
        evaluated_prefixes = sorted(set(prefix_seconds) | benchmark_prefixes)
        for case in cases:
            duration = case.audio.size / sampling_rate
            for prefix_s in evaluated_prefixes:
                if prefix_s >= duration:
                    continue
                for append in append_ms:
                    end_s = min(duration, prefix_s + append / 1000.0)
                    if end_s <= prefix_s:
                        continue
                    x_audio = case.audio[: int(round(prefix_s * sampling_rate))]
                    y_audio = case.audio[: int(round(end_s * sampling_rate))]
                    features_x, length_x = _processor_features(processor, x_audio)
                    features_y, length_y = _processor_features(processor, y_audio)
                    benchmark = any(
                        abs(end_s - item) < 0.011 for item in benchmark_seconds
                    )
                    callback = EncoderProbeCall(
                        features_x=features_x,
                        features_y=features_y,
                        feature_length_x=length_x,
                        feature_length_y=length_y,
                        sampling_rate=sampling_rate,
                        hop_length=hop_length,
                        trace_layers=args.trace_layers,
                        warmup_iterations=args.warmup_iterations if benchmark else 0,
                        timing_iterations=args.timing_iterations if benchmark else 0,
                    )
                    result = llm.apply_model(callback)[0]
                    geometry = result["geometry"]
                    stable_frames = result["stable_feature_frames"]
                    feature_metrics = tensor_metrics(
                        features_x[:, :stable_frames], features_y[:, :stable_frames]
                    )
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "mode": mode,
                        "case_id": case.case_id,
                        "source": case.source,
                        "language": case.language,
                        "request_kind": case.request_kind,
                        "prefix_seconds": prefix_s,
                        "end_seconds": end_s,
                        "append_ms": append,
                        "feature_length_x": length_x,
                        "feature_length_y": length_y,
                        "stable_feature_frames": stable_frames,
                        "stable_encoder_tokens": result["stable_encoder_tokens"],
                        "tail_feature_frames": result["tail_feature_frames"],
                        "attention_sequence_lengths_x": result[
                            "attention_sequence_lengths_x"
                        ],
                        "attention_sequence_lengths_y": result[
                            "attention_sequence_lengths_y"
                        ],
                        "cache_bytes": result["cache_bytes"],
                        "stable_feature_reuse_ratio": (
                            stable_frames / length_y if length_y else 0.0
                        ),
                        "stable_encoder_token_reuse_ratio": (
                            result["stable_encoder_tokens"]
                            / encoder_output_length(length_y)
                            if length_y
                            else 0.0
                        ),
                        "device_memory": result["device_memory"],
                        "feature_temporal": feature_metrics,
                        "encoder_temporal": result["temporal"],
                        "encoder_reconstruction": result["reconstruction"],
                        "repeat_noise": result["repeat_noise"],
                        "trace": result["trace"],
                    }
                    comparison_key = (
                        case.case_id,
                        round(prefix_s, 3),
                        round(end_s, 3),
                        append,
                    )
                    if benchmark and mode == "eager":
                        eager_embeddings[comparison_key] = result["full_embedding"]
                    elif (
                        benchmark
                        and mode == "graph"
                        and comparison_key in eager_embeddings
                    ):
                        cross_metrics = tensor_metrics(
                            eager_embeddings[comparison_key],
                            result["incremental_embedding"],
                        )
                        row["eager_to_graph_incremental"] = cross_metrics
                        cross_mode_rows.append(cross_metrics)
                    encoder_rows.append(row)
                    _append_jsonl(args.output_dir / "encoder_cases.jsonl", row)

                    timing = result["timing"]
                    if timing["iterations"]:
                        timing_row = {
                            "schema_version": SCHEMA_VERSION,
                            "mode": mode,
                            "case_id": case.case_id,
                            "end_seconds": end_s,
                            "stable_encoder_tokens": result["stable_encoder_tokens"],
                            "cache_bytes": result["cache_bytes"],
                            **timing,
                        }
                        timing_rows.append(timing_row)
                        _append_jsonl(args.output_dir / "timings.jsonl", timing_row)

                    if not args.skip_decoder:
                        prompt = case.prompt or args.prompt
                        baseline = _decoder_output(
                            llm, prompt, result["full_embedding"], args.max_tokens
                        )
                        candidate = _decoder_output(
                            llm,
                            prompt,
                            result["incremental_embedding"],
                            args.max_tokens,
                        )
                        decoder_row = {
                            "schema_version": SCHEMA_VERSION,
                            "mode": mode,
                            "case_id": case.case_id,
                            "request_kind": case.request_kind,
                            "end_seconds": end_s,
                            "token_ids_equal": baseline["token_ids"]
                            == candidate["token_ids"],
                            "output_nonempty": bool(baseline["token_ids"])
                            and bool(candidate["token_ids"]),
                            "text_equal": baseline["text"] == candidate["text"],
                            "baseline_token_ids": baseline["token_ids"],
                            "candidate_token_ids": candidate["token_ids"],
                            "first_token_top_id_equal": baseline[
                                "first_token_top_id"
                            ]
                            == candidate["first_token_top_id"],
                            "baseline_first_token_top_id": baseline[
                                "first_token_top_id"
                            ],
                            "candidate_first_token_top_id": candidate[
                                "first_token_top_id"
                            ],
                            "baseline_first_token_logprobs": baseline[
                                "first_token_logprobs"
                            ],
                            "candidate_first_token_logprobs": candidate[
                                "first_token_logprobs"
                            ],
                            "baseline_text": baseline["text"],
                            "candidate_text": candidate["text"],
                            "full_embedding_decode_ms": baseline["duration_ms"],
                            "incremental_embedding_decode_ms": candidate[
                                "duration_ms"
                            ],
                        }
                        if benchmark:
                            raw_audio = case.audio[
                                : int(round(end_s * sampling_rate))
                            ]
                            raw_output = _raw_audio_output(
                                llm, prompt, raw_audio, args.max_tokens
                            )
                            incremental_encoder_ms = timing[
                                "incremental_encoder_ms"
                            ]["p50"]
                            decoder_row.update(
                                {
                                    "raw_audio_token_ids_equal": raw_output[
                                        "token_ids"
                                    ]
                                    == baseline["token_ids"],
                                    "raw_audio_text_equal": raw_output["text"]
                                    == baseline["text"],
                                    "raw_audio_done_ms": raw_output["duration_ms"],
                                    "incremental_done_lower_bound_ms": (
                                        candidate["duration_ms"]
                                        + incremental_encoder_ms
                                        if incremental_encoder_ms is not None
                                        else None
                                    ),
                                }
                            )
                        if benchmark and mode == "eager":
                            eager_decoder_outputs[comparison_key] = baseline
                        elif (
                            benchmark
                            and mode == "graph"
                            and comparison_key in eager_decoder_outputs
                        ):
                            eager_baseline = eager_decoder_outputs[comparison_key]
                            decoder_row["eager_to_graph_token_ids_equal"] = (
                                eager_baseline["token_ids"]
                                == candidate["token_ids"]
                            )
                            decoder_row["eager_to_graph_text_equal"] = (
                                eager_baseline["text"] == candidate["text"]
                            )
                        decoder_rows.append(decoder_row)
                        _append_jsonl(
                            args.output_dir / "decoder_cases.jsonl", decoder_row
                        )
        del llm
        if len(modes) > 1:
            cleanup_dist_env_and_memory()

    feature_pass = all(row["feature_temporal"]["allclose"] for row in encoder_rows)
    temporal_pass = all(
        passes_numerical_gate(
            row["encoder_temporal"], row["repeat_noise"]
        )
        for row in encoder_rows
    )
    reconstruction_pass = all(
        passes_numerical_gate(
            row["encoder_reconstruction"], row["repeat_noise"]
        )
        for row in encoder_rows
    )
    decoder_pass = bool(decoder_rows) and all(
        row["token_ids_equal"]
        and row["text_equal"]
        and row["first_token_top_id_equal"]
        and row["output_nonempty"]
        for row in decoder_rows
    )
    raw_decoder_rows = [
        row for row in decoder_rows if "raw_audio_token_ids_equal" in row
    ]
    raw_decoder_pass = not raw_decoder_rows or all(
        row["raw_audio_token_ids_equal"] and row["raw_audio_text_equal"]
        for row in raw_decoder_rows
    )
    timed_reductions = [
        row["compute_reduction_ratio"]
        for row in timing_rows
        if row["compute_reduction_ratio"] is not None and row["end_seconds"] >= 8
    ]
    performance_worthy = (
        not timed_reductions or percentile(timed_reductions, 50) >= 0.35
    )
    cross_mode_pass = not cross_mode_rows or all(
        row["allclose"] and row["cosine"] >= 0.9999 for row in cross_mode_rows
    )
    cross_mode_decoder_rows = [
        row
        for row in decoder_rows
        if "eager_to_graph_token_ids_equal" in row
    ]
    cross_mode_decoder_pass = not cross_mode_decoder_rows or all(
        row["eager_to_graph_token_ids_equal"]
        and row["eager_to_graph_text_equal"]
        for row in cross_mode_decoder_rows
    )

    if not feature_pass:
        conclusion = "incremental_feature_extractor_required"
    elif not temporal_pass and any(
        row.get("trace", {}).get("conv_out", {}).get("allclose")
        for row in encoder_rows
    ):
        conclusion = "conv_only_reuse_feasible"
    elif (
        not temporal_pass
        or not reconstruction_pass
        or not cross_mode_pass
        or not cross_mode_decoder_pass
        or not raw_decoder_pass
        or (not args.skip_decoder and not decoder_pass)
    ):
        conclusion = "not_numerically_equivalent"
    elif not performance_worthy:
        conclusion = "equivalent_but_not_performance_worthy"
    else:
        conclusion = "full_window_reuse_feasible"

    max_cache_bytes = max(
        (row["cache_bytes"] for row in encoder_rows), default=0
    )
    free_memory_values = [
        row["device_memory"]["free_bytes"]
        for row in encoder_rows
        if row["device_memory"].get("free_bytes") is not None
    ]
    estimated_safe_sessions = (
        int(min(free_memory_values) * 0.8 // max_cache_bytes)
        if free_memory_values and max_cache_bytes
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "conclusion": conclusion,
        "geometry": geometry,
        "encoder_case_count": len(encoder_rows),
        "decoder_case_count": len(decoder_rows),
        "feature_temporal_pass": feature_pass,
        "encoder_temporal_pass": temporal_pass,
        "encoder_reconstruction_pass": reconstruction_pass,
        "decoder_exact_match_pass": decoder_pass if not args.skip_decoder else None,
        "raw_audio_to_embedding_decoder_pass": raw_decoder_pass,
        "eager_to_graph_incremental_pass": cross_mode_pass,
        "eager_to_graph_decoder_pass": cross_mode_decoder_pass,
        "decoder_validation_skipped": args.skip_decoder,
        "timing": summarize_timings(
            [
                {
                    "full_encoder_ms": row["full_encoder_ms"]["p50"],
                    "incremental_encoder_ms": row["incremental_encoder_ms"]["p50"],
                    "compute_reduction_ratio": row["compute_reduction_ratio"],
                }
                for row in timing_rows
                if row["full_encoder_ms"]["p50"] is not None
            ]
        ),
        "max_cache_bytes_per_case": max_cache_bytes,
        "five_to_six_second_feature_reuse_ratio": {
            "p50": percentile(
                [
                    row["stable_feature_reuse_ratio"]
                    for row in encoder_rows
                    if 5.0 <= row["end_seconds"] <= 6.0
                ],
                50,
            ),
            "p95": percentile(
                [
                    row["stable_feature_reuse_ratio"]
                    for row in encoder_rows
                    if 5.0 <= row["end_seconds"] <= 6.0
                ],
                95,
            ),
        },
        "cache_bytes_at_concurrency": {
            "20": 20
            * max_cache_bytes,
            "32": 32
            * max_cache_bytes,
        },
        "estimated_safe_session_upper_bound_at_80pct_free_memory": (
            estimated_safe_sessions
        ),
        "production_cache_implemented": False,
        "production_concurrency_benchmark_required": True,
        "limitations": [
            "This probe measures model-level cache-hit equivalence and encoder timing.",
            "Final p95 at concurrency 20/32 requires an opt-in end-to-end "
            "cache prototype.",
        ],
    }
    config["geometry"] = geometry
    _write_json(args.output_dir / "config.json", config)
    _write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "transcripts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fields = [
            "mode",
            "case_id",
            "end_seconds",
            "token_ids_equal",
            "text_equal",
            "baseline_text",
            "candidate_text",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in decoder_rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    run(_build_parser().parse_args())


if __name__ == "__main__":
    main()
