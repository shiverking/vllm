# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simulate streaming transcription over the REST transcription API.

Before running this script, start a vLLM server with a model that supports
``response_prefix`` in the audio transcription prompt, for example:

    vllm serve Qwen/Qwen3-ASR-1.7B

The client keeps streaming state locally:
1. Accumulate audio as if chunks were arriving from a microphone.
2. Send the growing audio buffer to ``/v1/audio/transcriptions``.
3. Pass stable transcript text back as ``response_prefix``.
4. Hold back a small tail so the model can revise chunk-boundary text.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio


@dataclass
class PrefixStreamingState:
    stable_text: str = ""
    unstable_text: str = ""
    last_printed_text: str = ""
    ttft_ms: float | None = None

    @property
    def display_text(self) -> str:
        return self.stable_text + self.unstable_text


@dataclass
class RequestResult:
    text: str
    latency_ms: float
    ttft_ms: float | None
    upload_audio_ms: float
    upload_bytes: int


def audio_to_wav_buffer(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    buffer.seek(0)
    buffer.name = "audio.wav"
    return buffer


def cap_text_by_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text
    return " ".join(words[-max_words:])


def split_with_holdback(text: str, holdback_words: int) -> tuple[str, str]:
    if holdback_words <= 0:
        return text, ""

    words = text.split()
    if len(words) <= holdback_words:
        return "", text

    stable = " ".join(words[:-holdback_words])
    unstable = " ".join(words[-holdback_words:])
    return stable, unstable


def append_with_spacing(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if prefix[-1].isspace() or suffix[0].isspace():
        return prefix + suffix
    return prefix + " " + suffix


def common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def merge_prefix_and_response(prefix: str, response_text: str) -> str:
    response_text = response_text or ""
    if not prefix:
        return response_text

    if response_text.startswith(prefix):
        return response_text

    if prefix.endswith(response_text):
        return prefix

    overlap_limit = min(len(prefix), len(response_text))
    for overlap in range(overlap_limit, 0, -1):
        if prefix[-overlap:] == response_text[:overlap]:
            return prefix + response_text[overlap:]

    return append_with_spacing(prefix, response_text)


def merge_history_and_candidate(history: str, candidate_text: str) -> str:
    candidate_text = candidate_text or ""
    if not history:
        return candidate_text
    if not candidate_text:
        return history

    if candidate_text.startswith(history):
        return candidate_text

    if history.endswith(candidate_text):
        return history

    overlap_limit = min(len(history), len(candidate_text))
    for overlap in range(overlap_limit, 0, -1):
        if history[-overlap:] == candidate_text[:overlap]:
            return history + candidate_text[overlap:]

    return append_with_spacing(history, candidate_text)


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def print_delta(state: PrefixStreamingState) -> bool:
    display_text = state.display_text
    delta_start = common_prefix_len(state.last_printed_text, display_text)
    delta = display_text[delta_start:]
    if delta:
        print(delta, end="", flush=True)
    state.last_printed_text = display_text
    return bool(delta)


def build_audio_endpoint_url(api_base: str, endpoint: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/audio"):
        return f"{base}/{endpoint}"
    return f"{base}/audio/{endpoint}"


def extract_stream_delta(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if content is None:
        return ""
    return content


def post_audio_request(
    *,
    api_base: str,
    endpoint: str,
    audio: np.ndarray,
    sample_rate: int,
    model: str,
    language: str | None,
    to_language: str | None,
    response_prefix: str,
    stream: bool,
    temperature: float,
    timeout: float,
) -> RequestResult:
    data: dict[str, Any] = {
        "model": model,
        "response_format": "json",
        "response_prefix": response_prefix,
        "stream": "true" if stream else "false",
        "temperature": temperature,
    }
    if language:
        data["language"] = language
    if to_language:
        data["to_language"] = to_language

    wav_buffer = audio_to_wav_buffer(audio, sample_rate)
    upload_bytes = wav_buffer.getbuffer().nbytes
    files = {"file": ("audio.wav", wav_buffer, "audio/wav")}
    request_start = time.perf_counter()
    response = requests.post(
        build_audio_endpoint_url(api_base, endpoint),
        data=data,
        files=files,
        stream=stream,
        timeout=timeout,
    )
    response.raise_for_status()

    ttft_ms: float | None = None
    if stream:
        text_parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            if line.strip() == "[DONE]":
                break

            payload = json.loads(line)
            delta = extract_stream_delta(payload)
            if delta:
                if ttft_ms is None:
                    ttft_ms = seconds_to_ms(time.perf_counter() - request_start)
                text_parts.append(delta)

        text = "".join(text_parts)
    else:
        text = response.json()["text"]

    latency_ms = seconds_to_ms(time.perf_counter() - request_start)
    return RequestResult(
        text=text,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        upload_audio_ms=seconds_to_ms(len(audio) / sample_rate),
        upload_bytes=upload_bytes,
    )


def run_prefix_streaming(args: argparse.Namespace) -> None:
    e2e_start = time.perf_counter()
    audio_path = args.audio_path
    if audio_path is None:
        audio_path = str(AudioAsset("mary_had_lamb").get_local_path())
    audio, sample_rate = load_audio(audio_path, sr=args.sample_rate, mono=True)
    chunk_samples = max(1, int(args.chunk_seconds * sample_rate))
    max_window_samples = (
        int(args.max_audio_window_seconds * sample_rate)
        if args.max_audio_window_seconds > 0
        else None
    )
    state = PrefixStreamingState()
    endpoint = "translations" if args.task == "translate" else "transcriptions"
    total_uploaded_audio_ms = 0.0
    total_uploaded_bytes = 0
    request_latencies_ms: list[float] = []
    first_audio_request_start: float | None = None
    round_idx = 0

    audio_duration_s = len(audio) / sample_rate
    print(
        "[setup] "
        f"audio={audio_path} duration_ms={seconds_to_ms(audio_duration_s):.0f} "
        f"sample_rate={sample_rate} "
        f"chunk_ms={seconds_to_ms(args.chunk_seconds):.0f} "
        f"stream={args.stream} "
        f"holdback_words={args.holdback_words} "
        f"max_prefix_words={args.max_prefix_words} "
        "max_audio_window_ms="
        f"{seconds_to_ms(args.max_audio_window_seconds):.0f}"
    )

    print("Transcription: ", end="", flush=True)
    for end in range(chunk_samples, len(audio) + chunk_samples, chunk_samples):
        round_idx += 1
        end = min(end, len(audio))
        start = 0
        if max_window_samples is not None:
            start = max(0, end - max_window_samples)

        prefix = cap_text_by_words(state.stable_text, args.max_prefix_words)
        prefix_words = len(prefix.split())
        audio_start_s = start / sample_rate
        audio_end_s = end / sample_rate
        print(
            "\n"
            f"[round {round_idx}] request "
            "audio_window_ms="
            f"{seconds_to_ms(audio_start_s):.0f}-{seconds_to_ms(audio_end_s):.0f} "
            f"upload_audio_ms={seconds_to_ms((end - start) / sample_rate):.0f} "
            f"prefix_words={prefix_words} "
            f"stable_words={len(state.stable_text.split())} "
            f"unstable_words={len(state.unstable_text.split())}",
            flush=True,
        )

        if first_audio_request_start is None:
            first_audio_request_start = time.perf_counter()
        result = post_audio_request(
            api_base=args.api_base,
            endpoint=endpoint,
            audio=audio[start:end],
            sample_rate=sample_rate,
            model=args.model,
            language=args.language,
            to_language=args.to_language,
            response_prefix=prefix,
            stream=args.stream,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        total_uploaded_audio_ms += result.upload_audio_ms
        total_uploaded_bytes += result.upload_bytes
        request_latencies_ms.append(result.latency_ms)
        candidate_text = merge_history_and_candidate(
            state.stable_text,
            merge_prefix_and_response(prefix, result.text),
        )
        stable_text, unstable_text = split_with_holdback(
            candidate_text,
            args.holdback_words,
        )
        state.stable_text = stable_text
        state.unstable_text = unstable_text
        has_display_text = bool(state.display_text)
        if state.ttft_ms is None and result.ttft_ms is not None:
            state.ttft_ms = result.ttft_ms
        elif state.ttft_ms is None and has_display_text:
            assert first_audio_request_start is not None
            state.ttft_ms = seconds_to_ms(
                time.perf_counter() - first_audio_request_start
            )
        print(
            f"[round {round_idx}] response "
            f"latency_ms={result.latency_ms:.0f} "
            f"ttft_ms={result.ttft_ms if result.ttft_ms is not None else -1:.0f} "
            f"upload_bytes={result.upload_bytes} "
            f"response_words={len(result.text.split())} "
            f"new_stable_words={len(state.stable_text.split())} "
            f"new_unstable_words={len(state.unstable_text.split())}",
            flush=True,
        )
        print("[delta] ", end="", flush=True)
        print_delta(state)
        print()

        if end >= len(audio):
            break

    final_prefix = cap_text_by_words(state.stable_text, args.max_prefix_words)
    final_audio = audio
    if max_window_samples is not None:
        final_audio = audio[-max_window_samples:]

    print(
        "\n"
        "[final] request "
        f"upload_audio_ms={seconds_to_ms(len(final_audio) / sample_rate):.0f} "
        f"prefix_words={len(final_prefix.split())}",
        flush=True,
    )
    final_result = post_audio_request(
        api_base=args.api_base,
        endpoint=endpoint,
        audio=final_audio,
        sample_rate=sample_rate,
        model=args.model,
        language=args.language,
        to_language=args.to_language,
        response_prefix=final_prefix,
        stream=args.stream,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    total_uploaded_audio_ms += final_result.upload_audio_ms
    total_uploaded_bytes += final_result.upload_bytes
    request_latencies_ms.append(final_result.latency_ms)
    state.stable_text = merge_history_and_candidate(
        state.stable_text,
        merge_prefix_and_response(final_prefix, final_result.text),
    )
    state.unstable_text = ""
    if state.ttft_ms is None and state.display_text:
        if final_result.ttft_ms is not None:
            state.ttft_ms = final_result.ttft_ms
        else:
            assert first_audio_request_start is not None
            state.ttft_ms = seconds_to_ms(
                time.perf_counter() - first_audio_request_start
            )
    print(
        "[final] response "
        f"latency_ms={final_result.latency_ms:.0f} "
        "ttft_ms="
        f"{final_result.ttft_ms if final_result.ttft_ms is not None else -1:.0f} "
        f"upload_bytes={final_result.upload_bytes} "
        f"response_words={len(final_result.text.split())}",
        flush=True,
    )
    print("[delta] ", end="", flush=True)
    print_delta(state)
    print()

    e2e_ms = seconds_to_ms(time.perf_counter() - e2e_start)
    avg_latency_ms = sum(request_latencies_ms) / len(request_latencies_ms)
    max_latency_ms = max(request_latencies_ms)
    print(f"\n\nFinal {args.task} result:\n{state.stable_text}")
    print(
        "\n[metrics] "
        f"ttft_ms={state.ttft_ms if state.ttft_ms is not None else -1:.0f} "
        f"e2e_ms={e2e_ms:.0f} "
        f"requests={len(request_latencies_ms)} "
        f"avg_request_latency_ms={avg_latency_ms:.0f} "
        f"max_request_latency_ms={max_latency_ms:.0f} "
        f"uploaded_audio_ms={total_uploaded_audio_ms:.0f} "
        f"uploaded_bytes={total_uploaded_bytes}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REST prefix-streaming transcription client for vLLM."
    )
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-ASR-1.7B")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--to-language", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--holdback-words", type=int, default=5)
    parser.add_argument("--max-prefix-words", type=int, default=100)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use the REST endpoint's SSE stream and record first-delta TTFT.",
    )
    parser.add_argument(
        "--max-audio-window-seconds",
        type=float,
        default=0.0,
        help="If > 0, send only the latest N seconds instead of the full buffer.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_prefix_streaming(parse_args())
