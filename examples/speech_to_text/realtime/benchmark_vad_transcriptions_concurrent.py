# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Concurrent streaming benchmark for /v1/audio/transcriptions.

The benchmark can either send each audio file as one streaming transcription
request, or split each file with the local fixed-head + Silero VAD pipeline and
send every segment as an independent streaming transcription request.
"""

import argparse
import io
import json
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
VAD_LOCK = threading.Lock()


@dataclass
class Segment:
    index: int
    audio: np.ndarray
    mode: str
    emit_offset_s: float


@dataclass
class StreamResult:
    ttft_ms: float
    tpot_ms: list[float]
    e2e_ms: float
    chunks: int
    chars: int
    saw_done: bool


@dataclass
class SegmentResult:
    index: int
    mode: str
    duration_s: float
    emit_offset_s: float
    request_start_offset_s: float
    stream: StreamResult


@dataclass
class PrecomputedSegment:
    index: int
    filename: str
    audio_bytes: bytes
    duration_s: float
    mode: str
    emit_offset_s: float


@dataclass
class PrecomputedRequest:
    segments: list[PrecomputedSegment]
    vad_init_ms: float
    vad_time_ms: float
    speech_duration_s: float


class SileroSpeechDetector:

    def __init__(
        self,
        *,
        sampling_rate: int,
        threshold: float,
        onnx: bool,
    ) -> None:
        if sampling_rate not in (8000, 16000):
            raise ValueError(
                "Silero VAD supports 8000 Hz and 16000 Hz audio, "
                f"got {sampling_rate} Hz."
            )
        self._sampling_rate = sampling_rate
        self._threshold = threshold
        self._frame_size = 256 if sampling_rate == 8000 else 512
        self._model = load_silero_model(onnx)
        reset_states = getattr(self._model, "reset_states", None)
        if reset_states is not None:
            reset_states()

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def is_speech(self, audio: np.ndarray) -> bool:
        import torch

        audio_f32 = np.asarray(audio, dtype=np.float32)
        with torch.inference_mode():
            speech_prob = self._model(
                torch.from_numpy(audio_f32),
                self._sampling_rate,
            ).item()
        return float(speech_prob) >= self._threshold


class RealtimeVADSegmenter:

    def __init__(
        self,
        detector: SileroSpeechDetector,
        *,
        sampling_rate: int,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        max_segment_duration_s: float,
    ) -> None:
        self._detector = detector
        self._frame_size = detector.frame_size
        self._min_speech_samples = int(
            sampling_rate * min_speech_duration_ms / 1000
        )
        self._min_silence_samples = int(
            sampling_rate * min_silence_duration_ms / 1000
        )
        self._speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        self._max_segment_samples = int(sampling_rate * max_segment_duration_s)

        self._pending_audio = np.empty(0, dtype=np.float32)
        self._pre_roll_audio = np.empty(0, dtype=np.float32)
        self._segment_chunks: list[np.ndarray] = []
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._segment_samples = 0

    def write_audio(self, audio: np.ndarray) -> list[np.ndarray]:
        audio_f32 = np.asarray(audio, dtype=np.float32)
        self._pending_audio = np.concatenate((self._pending_audio, audio_f32))

        segments: list[np.ndarray] = []
        while self._pending_audio.shape[-1] >= self._frame_size:
            frame = self._pending_audio[:self._frame_size].copy()
            self._pending_audio = self._pending_audio[self._frame_size:]

            if self._detector.is_speech(frame):
                self._append_speech_frame(frame)
            elif self._in_speech:
                self._append_silence_frame(frame)
            else:
                self._append_pre_roll(frame)

            if self._in_speech and self._silence_samples >= (
                self._min_silence_samples
            ):
                segment = self._flush_active_segment(
                    trim_trailing_silence=True
                )
                if segment is not None:
                    segments.append(segment)
            elif self._in_speech and self._segment_samples >= (
                self._max_segment_samples
            ):
                segment = self._flush_active_segment(
                    trim_trailing_silence=False
                )
                if segment is not None:
                    segments.append(segment)

        return segments

    def flush(self) -> np.ndarray | None:
        if self._pending_audio.shape[-1] > 0:
            if self._in_speech:
                self._segment_chunks.append(self._pending_audio.copy())
                self._segment_samples += self._pending_audio.shape[-1]
            self._pending_audio = np.empty(0, dtype=np.float32)

        return self._flush_active_segment(trim_trailing_silence=False)

    def _append_speech_frame(self, frame: np.ndarray) -> None:
        if not self._in_speech:
            self._in_speech = True
            if self._pre_roll_audio.shape[-1] > 0:
                self._segment_chunks.append(self._pre_roll_audio.copy())
                self._segment_samples += self._pre_roll_audio.shape[-1]
            self._pre_roll_audio = np.empty(0, dtype=np.float32)

        self._segment_chunks.append(frame)
        self._speech_samples += frame.shape[-1]
        self._segment_samples += frame.shape[-1]
        self._silence_samples = 0

    def _append_silence_frame(self, frame: np.ndarray) -> None:
        self._segment_chunks.append(frame)
        self._segment_samples += frame.shape[-1]
        self._silence_samples += frame.shape[-1]

    def _append_pre_roll(self, frame: np.ndarray) -> None:
        if self._speech_pad_samples <= 0:
            return
        self._pre_roll_audio = np.concatenate((self._pre_roll_audio, frame))
        if self._pre_roll_audio.shape[-1] > self._speech_pad_samples:
            self._pre_roll_audio = self._pre_roll_audio[
                -self._speech_pad_samples:
            ]

    def _flush_active_segment(
        self,
        *,
        trim_trailing_silence: bool,
    ) -> np.ndarray | None:
        if not self._in_speech:
            return None

        segment = np.concatenate(self._segment_chunks)
        if trim_trailing_silence:
            trim_samples = max(0, self._silence_samples - self._speech_pad_samples)
            if trim_samples > 0:
                segment = segment[:-trim_samples]

        should_emit = self._speech_samples >= self._min_speech_samples
        self._reset_active_segment()
        return segment if should_emit and segment.shape[-1] > 0 else None

    def _reset_active_segment(self) -> None:
        self._segment_chunks = []
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._segment_samples = 0


@cache
def load_silero_model(onnx: bool):
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=onnx)


def find_all_audio(audio_dir: str) -> list[Path]:
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")

    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        raise RuntimeError(f"no audio files found in {audio_dir}")
    return files


def get_audio_duration_seconds(audio_path: Path) -> float | None:
    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                rate = wav_file.getframerate()
                if rate > 0:
                    return wav_file.getnframes() / float(rate)
        except Exception:
            pass

    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio_path))
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        return None

    return None


def load_audio_16k(audio_path: Path) -> np.ndarray:
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        if audio_path.suffix.lower() != ".wav":
            raise ImportError(
                "Install soundfile to decode non-WAV audio without importing "
                "vLLM in the client process."
            )
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            raw_audio = wav_file.readframes(wav_file.getnframes())
        if sample_width == 2:
            audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(
                np.float32
            )
            audio_array /= 32768.0
        elif sample_width == 4:
            audio_array = np.frombuffer(raw_audio, dtype=np.int32).astype(
                np.float32
            )
            audio_array /= 2147483648.0
        elif sample_width == 1:
            audio_array = np.frombuffer(raw_audio, dtype=np.uint8).astype(
                np.float32
            )
            audio_array = (audio_array - 128.0) / 128.0
        else:
            raise ValueError(f"unsupported WAV sample width: {sample_width}")
        if channels > 1:
            audio_array = audio_array.reshape(-1, channels).mean(axis=1)
    else:
        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )
        audio_array = np.asarray(audio, dtype=np.float32)

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    if sample_rate == SAMPLE_RATE:
        return audio_array

    try:
        from scipy.signal import resample_poly  # type: ignore

        gcd = np.gcd(sample_rate, SAMPLE_RATE)
        resampled = resample_poly(
            audio_array,
            SAMPLE_RATE // gcd,
            sample_rate // gcd,
        )
        return np.asarray(resampled, dtype=np.float32)
    except ImportError:
        duration_s = len(audio_array) / float(sample_rate)
        output_samples = max(1, int(duration_s * SAMPLE_RATE))
        old_positions = np.linspace(0.0, duration_s, num=len(audio_array))
        new_positions = np.linspace(0.0, duration_s, num=output_samples)
        return np.interp(new_positions, old_positions, audio_array).astype(
            np.float32
        )


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(BYTES_PER_SAMPLE)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()


def parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None

    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return {"done": True}

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def extract_delta_text(event: dict[str, Any]) -> str:
    if not event or event.get("done"):
        return ""

    choices = event.get("choices") or []
    if not choices:
        return ""

    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def send_transcription_stream(
    *,
    url: str,
    model: str,
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    language: str | None,
    max_completion_tokens: int,
) -> StreamResult:
    data = {
        "model": model,
        "response_format": "json",
        "stream": "true",
        "temperature": "0.0",
        "max_completion_tokens": str(max_completion_tokens),
    }
    if language:
        data["language"] = language

    start_time = time.perf_counter()
    first_text_time = None
    last_text_time = None
    ttft_ms = None
    tpot_ms: list[float] = []
    chunks = 0
    chars = 0
    saw_done = False

    files = {"file": (filename, audio_bytes, content_type)}
    with requests.post(
        url,
        data=data,
        files=files,
        stream=True,
        timeout=600,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                "request failed, status_code="
                f"{response.status_code}, body={response.text}"
            )

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            event = parse_sse_line(raw_line)
            if event is None:
                continue
            if event.get("done"):
                saw_done = True
                break

            text = extract_delta_text(event)
            if not text:
                continue

            now = time.perf_counter()
            chunks += 1
            chars += len(text)
            if first_text_time is None:
                first_text_time = now
                ttft_ms = (first_text_time - start_time) * 1000
            else:
                tpot_ms.append((now - last_text_time) * 1000)
            last_text_time = now

    e2e_ms = (time.perf_counter() - start_time) * 1000
    if ttft_ms is None or chunks == 0:
        raise RuntimeError("no streamed text received, TTFT is unavailable")

    return StreamResult(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        chunks=chunks,
        chars=chars,
        saw_done=saw_done,
    )


def create_segmenter(args: argparse.Namespace) -> RealtimeVADSegmenter:
    detector = SileroSpeechDetector(
        sampling_rate=SAMPLE_RATE,
        threshold=args.vad_threshold,
        onnx=args.vad_onnx,
    )
    return RealtimeVADSegmenter(
        detector,
        sampling_rate=SAMPLE_RATE,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        max_segment_duration_s=args.vad_max_segment_s,
    )


def emit_fixed_head(
    audio: np.ndarray,
    *,
    args: argparse.Namespace,
    job_start_time: float,
) -> tuple[list[Segment], int]:
    head_samples = int(SAMPLE_RATE * args.head_fixed_segment_s)
    if head_samples <= 0:
        return [], 0

    head_samples = min(head_samples, len(audio))
    if head_samples <= 0:
        return [], 0

    return [
        Segment(
            index=1,
            audio=audio[:head_samples],
            mode="fixed_head",
            emit_offset_s=time.perf_counter() - job_start_time,
        )
    ], head_samples


def transcribe_segment(
    segment: Segment,
    *,
    url: str,
    args: argparse.Namespace,
    job_start_time: float,
) -> SegmentResult:
    request_start_offset_s = time.perf_counter() - job_start_time
    stream = send_transcription_stream(
        url=url,
        model=args.model,
        audio_bytes=audio_to_wav_bytes(segment.audio),
        filename=f"segment_{segment.index}.wav",
        content_type="audio/wav",
        language=args.language,
        max_completion_tokens=args.segment_max_completion_tokens,
    )
    return SegmentResult(
        index=segment.index,
        mode=segment.mode,
        duration_s=len(segment.audio) / SAMPLE_RATE,
        emit_offset_s=segment.emit_offset_s,
        request_start_offset_s=request_start_offset_s,
        stream=stream,
    )


def transcribe_precomputed_segment(
    segment: PrecomputedSegment,
    *,
    url: str,
    args: argparse.Namespace,
    job_start_time: float,
) -> SegmentResult:
    request_start_offset_s = time.perf_counter() - job_start_time
    stream = send_transcription_stream(
        url=url,
        model=args.model,
        audio_bytes=segment.audio_bytes,
        filename=segment.filename,
        content_type="audio/wav",
        language=args.language,
        max_completion_tokens=args.segment_max_completion_tokens,
    )
    return SegmentResult(
        index=segment.index,
        mode=segment.mode,
        duration_s=segment.duration_s,
        emit_offset_s=segment.emit_offset_s,
        request_start_offset_s=request_start_offset_s,
        stream=stream,
    )


def run_full_audio_request(
    audio_path: Path,
    *,
    audio_duration_s: float | None,
    url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    with open(audio_path, "rb") as audio_file:
        stream = send_transcription_stream(
            url=url,
            model=args.model,
            audio_bytes=audio_file.read(),
            filename=audio_path.name,
            content_type="application/octet-stream",
            language=args.language,
            max_completion_tokens=args.max_completion_tokens,
        )

    rtf = None
    if audio_duration_s is not None and audio_duration_s > 0:
        rtf = (stream.e2e_ms / 1000.0) / audio_duration_s

    return {
        "ttft_ms": stream.ttft_ms,
        "e2e_ms": stream.e2e_ms,
        "tpot_ms": stream.tpot_ms,
        "rtf": rtf,
        "streamed_chunks": stream.chunks,
        "chars": stream.chars,
        "saw_done": stream.saw_done,
        "segments": 1,
        "vad_init_ms": 0.0,
        "vad_time_ms": 0.0,
        "speech_duration_s": audio_duration_s,
    }


def build_vad_metrics(
    *,
    segment_results: list[SegmentResult],
    audio_duration_s: float | None,
    vad_init_ms: float,
    vad_time_ms: float,
    speech_duration_s: float,
    precomputed: bool,
) -> dict[str, Any]:
    if not segment_results:
        raise RuntimeError("no transcription segments were produced")

    segment_results.sort(key=lambda item: item.index)
    first_token = min(
        segment_results,
        key=lambda item: item.request_start_offset_s + item.stream.ttft_ms / 1000,
    )
    ttft_s = first_token.request_start_offset_s + first_token.stream.ttft_ms / 1000
    e2e_s = max(
        item.request_start_offset_s + item.stream.e2e_ms / 1000
        for item in segment_results
    )
    rtf = None
    if audio_duration_s is not None and audio_duration_s > 0:
        rtf = e2e_s / audio_duration_s

    return {
        "ttft_ms": ttft_s * 1000,
        "e2e_ms": e2e_s * 1000,
        "tpot_ms": [
            value for item in segment_results for value in item.stream.tpot_ms
        ],
        "rtf": rtf,
        "streamed_chunks": sum(item.stream.chunks for item in segment_results),
        "chars": sum(item.stream.chars for item in segment_results),
        "saw_done": all(item.stream.saw_done for item in segment_results),
        "segments": len(segment_results),
        "vad_init_ms": vad_init_ms,
        "vad_time_ms": vad_time_ms,
        "speech_duration_s": speech_duration_s,
        "precomputed": precomputed,
        "first_segment_emit_ms": segment_results[0].emit_offset_s * 1000,
        "first_request_start_ms": (
            min(item.request_start_offset_s for item in segment_results) * 1000
        ),
        "first_request_ttft_ms": first_token.stream.ttft_ms,
        "first_token_segment": first_token.index,
        "segment_results": [
            {
                "index": item.index,
                "mode": item.mode,
                "duration_s": item.duration_s,
                "emit_ms": item.emit_offset_s * 1000,
                "request_start_ms": item.request_start_offset_s * 1000,
                "ttft_ms": item.stream.ttft_ms,
                "e2e_ms": item.stream.e2e_ms,
                "chunks": item.stream.chunks,
                "chars": item.stream.chars,
            }
            for item in segment_results
        ],
    }


def run_precomputed_vad_request(
    precomputed: PrecomputedRequest,
    *,
    audio_duration_s: float | None,
    url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    job_start_time = time.perf_counter()
    segment_results: list[SegmentResult] = []
    with ThreadPoolExecutor(max_workers=args.segment_concurrency) as executor:
        futures = [
            executor.submit(
                transcribe_precomputed_segment,
                segment,
                url=url,
                args=args,
                job_start_time=job_start_time,
            )
            for segment in precomputed.segments
        ]
        for future in as_completed(futures):
            segment_results.append(future.result())

    return build_vad_metrics(
        segment_results=segment_results,
        audio_duration_s=audio_duration_s,
        vad_init_ms=precomputed.vad_init_ms,
        vad_time_ms=precomputed.vad_time_ms,
        speech_duration_s=precomputed.speech_duration_s,
        precomputed=True,
    )


def run_vad_audio_request(
    audio_path: Path,
    *,
    audio_duration_s: float | None,
    url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    audio = load_audio_16k(audio_path)
    job_start_time = time.perf_counter()
    fixed_segments, audio_offset = emit_fixed_head(
        audio,
        args=args,
        job_start_time=job_start_time,
    )

    segment_results: list[SegmentResult] = []
    with ThreadPoolExecutor(max_workers=args.segment_concurrency) as executor:
        futures = [
            executor.submit(
                transcribe_segment,
                segment,
                url=url,
                args=args,
                job_start_time=job_start_time,
            )
            for segment in fixed_segments
        ]

        next_index = len(fixed_segments) + 1
        vad_init_s = 0.0
        vad_time_s = 0.0
        if audio_offset < len(audio):
            # Silero/PyTorch native backends can crash the client process when
            # several benchmark threads call the local VAD at the same time.
            # Segment ASR requests still run concurrently; only local VAD scans
            # are serialized.
            with VAD_LOCK:
                vad_init_start = time.perf_counter()
                segmenter = create_segmenter(args)
                vad_init_s = time.perf_counter() - vad_init_start
                vad_scan_start = time.perf_counter()
                chunk_samples = max(
                    1, int(SAMPLE_RATE * args.vad_chunk_duration_ms / 1000)
                )

                for start in range(audio_offset, len(audio), chunk_samples):
                    chunk = audio[start:start + chunk_samples]
                    for segment_audio in segmenter.write_audio(chunk):
                        segment = Segment(
                            index=next_index,
                            audio=segment_audio,
                            mode="silero",
                            emit_offset_s=(
                                time.perf_counter() - job_start_time
                            ),
                        )
                        futures.append(
                            executor.submit(
                                transcribe_segment,
                                segment,
                                url=url,
                                args=args,
                                job_start_time=job_start_time,
                            )
                        )
                        next_index += 1

                remaining = segmenter.flush()
                if remaining is not None and len(remaining) > 0:
                    segment = Segment(
                        index=next_index,
                        audio=remaining,
                        mode="silero",
                        emit_offset_s=time.perf_counter() - job_start_time,
                    )
                    futures.append(
                        executor.submit(
                            transcribe_segment,
                            segment,
                            url=url,
                            args=args,
                            job_start_time=job_start_time,
                        )
                    )

                vad_time_s = time.perf_counter() - vad_scan_start

        for future in as_completed(futures):
            segment_results.append(future.result())

    speech_duration_s = sum(item.duration_s for item in segment_results)
    return build_vad_metrics(
        segment_results=segment_results,
        audio_duration_s=audio_duration_s,
        vad_init_ms=vad_init_s * 1000,
        vad_time_ms=vad_time_s * 1000,
        speech_duration_s=speech_duration_s,
        precomputed=False,
    )


def precompute_vad_request(
    audio_path: Path,
    *,
    args: argparse.Namespace,
) -> PrecomputedRequest:
    audio = load_audio_16k(audio_path)
    job_start_time = time.perf_counter()
    fixed_segments, audio_offset = emit_fixed_head(
        audio,
        args=args,
        job_start_time=job_start_time,
    )

    segments: list[Segment] = list(fixed_segments)
    next_index = len(segments) + 1
    vad_init_s = 0.0
    vad_time_s = 0.0
    if audio_offset < len(audio):
        vad_init_start = time.perf_counter()
        segmenter = create_segmenter(args)
        vad_init_s = time.perf_counter() - vad_init_start
        vad_scan_start = time.perf_counter()
        chunk_samples = max(
            1, int(SAMPLE_RATE * args.vad_chunk_duration_ms / 1000)
        )

        for start in range(audio_offset, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            for segment_audio in segmenter.write_audio(chunk):
                segments.append(
                    Segment(
                        index=next_index,
                        audio=segment_audio,
                        mode="silero",
                        emit_offset_s=time.perf_counter() - job_start_time,
                    )
                )
                next_index += 1

        remaining = segmenter.flush()
        if remaining is not None and len(remaining) > 0:
            segments.append(
                Segment(
                    index=next_index,
                    audio=remaining,
                    mode="silero",
                    emit_offset_s=time.perf_counter() - job_start_time,
                )
            )

        vad_time_s = time.perf_counter() - vad_scan_start

    if not segments:
        raise RuntimeError(f"no VAD segments produced for {audio_path}")

    precomputed_segments = [
        PrecomputedSegment(
            index=segment.index,
            filename=f"{audio_path.stem}_segment_{segment.index}.wav",
            audio_bytes=audio_to_wav_bytes(segment.audio),
            duration_s=len(segment.audio) / SAMPLE_RATE,
            mode=segment.mode,
            emit_offset_s=segment.emit_offset_s,
        )
        for segment in segments
    ]
    return PrecomputedRequest(
        segments=precomputed_segments,
        vad_init_ms=vad_init_s * 1000,
        vad_time_ms=vad_time_s * 1000,
        speech_duration_s=sum(item.duration_s for item in precomputed_segments),
    )


def run_one_request(
    request_id: int,
    audio_path: Path,
    audio_duration_s: float | None,
    url: str,
    args: argparse.Namespace,
    start_event: threading.Event,
    precomputed: PrecomputedRequest | None = None,
) -> dict[str, Any]:
    start_event.wait()
    try:
        if args.mode == "full":
            metrics = run_full_audio_request(
                audio_path,
                audio_duration_s=audio_duration_s,
                url=url,
                args=args,
            )
        elif precomputed is not None:
            metrics = run_precomputed_vad_request(
                precomputed,
                audio_duration_s=audio_duration_s,
                url=url,
                args=args,
            )
        else:
            metrics = run_vad_audio_request(
                audio_path,
                audio_duration_s=audio_duration_s,
                url=url,
                args=args,
            )

        return {
            "ok": True,
            "request_id": request_id,
            "file": audio_path.name,
            "audio_path": str(audio_path.resolve()),
            "audio_duration_s": audio_duration_s,
            "error": None,
            **metrics,
        }
    except Exception as exc:
        return {
            "ok": False,
            "request_id": request_id,
            "file": audio_path.name,
            "audio_path": str(audio_path.resolve()),
            "audio_duration_s": audio_duration_s,
            "error": str(exc),
            "ttft_ms": None,
            "e2e_ms": None,
            "tpot_ms": [],
            "rtf": None,
            "segments": 0,
        }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def fmt(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def print_metric(label: str, value: float | None, precision: int = 2) -> None:
    print(f"{label:<42}{fmt(value, precision):>10}")


def print_int_metric(label: str, value: int) -> None:
    print(f"{label:<42}{value:>10}")


def print_summary(
    *,
    results: list[dict[str, Any]],
    concurrency: int,
    benchmark_duration_s: float,
) -> dict[str, Any]:
    success = [item for item in results if item["ok"]]
    failed = [item for item in results if not item["ok"]]
    ttft_values = [item["ttft_ms"] for item in success]
    e2e_values = [item["e2e_ms"] for item in success]
    tpot_values = [value for item in success for value in item["tpot_ms"]]
    rtf_values = [item["rtf"] for item in success if item["rtf"] is not None]
    segment_values = [item["segments"] for item in success]
    vad_time_values = [item.get("vad_time_ms", 0.0) for item in success]
    audio_durations = [
        item["audio_duration_s"] for item in success
        if item["audio_duration_s"] is not None
    ]
    total_audio_duration_s = float(sum(audio_durations))
    request_throughput = (
        len(success) / benchmark_duration_s if benchmark_duration_s > 0 else 0.0
    )
    audio_throughput = (
        total_audio_duration_s / benchmark_duration_s
        if benchmark_duration_s > 0 else 0.0
    )

    print("=" * 56)
    print("              Transcription Benchmark             ")
    print("=" * 56)
    print_int_metric("Successful requests:", len(success))
    print_int_metric("Failed requests:", len(failed))
    print_int_metric("Maximum request concurrency:", concurrency)
    print_metric("Benchmark duration (s):", benchmark_duration_s, 2)
    print_metric("Request throughput (req/s):", request_throughput, 2)
    print_metric("Audio throughput (audio s/s):", audio_throughput, 2)
    print("-" * 56)
    print_metric("Mean TTFT (ms):", mean(ttft_values), 2)
    print_metric("Median TTFT (ms):", median(ttft_values), 2)
    print_metric("P90 TTFT (ms):", percentile(ttft_values, 90), 2)
    print_metric("P95 TTFT (ms):", percentile(ttft_values, 95), 2)
    print_metric("P99 TTFT (ms):", percentile(ttft_values, 99), 2)
    print("-" * 56)
    print_metric("Mean E2E (ms):", mean(e2e_values), 2)
    print_metric("Median E2E (ms):", median(e2e_values), 2)
    print_metric("P90 E2E (ms):", percentile(e2e_values, 90), 2)
    print_metric("P95 E2E (ms):", percentile(e2e_values, 95), 2)
    print_metric("P99 E2E (ms):", percentile(e2e_values, 99), 2)
    print("-" * 56)
    print_metric("Mean TPOT (ms):", mean(tpot_values), 2)
    print_metric("P99 TPOT (ms):", percentile(tpot_values, 99), 2)
    print_metric("Mean RTF:", mean(rtf_values), 3)
    print_metric("P99 RTF:", percentile(rtf_values, 99), 3)
    print("-" * 56)
    print_metric("Mean segments/request:", mean(segment_values), 2)
    print_metric("Mean VAD time (ms):", mean(vad_time_values), 2)
    print("=" * 56)

    return {
        "successful_requests": len(success),
        "failed_requests": len(failed),
        "benchmark_duration_s": benchmark_duration_s,
        "request_throughput_req_s": request_throughput,
        "audio_duration_total_s": total_audio_duration_s,
        "audio_throughput_s_per_s": audio_throughput,
        "ttft_ms": {
            "mean": mean(ttft_values),
            "median": median(ttft_values),
            "p90": percentile(ttft_values, 90),
            "p95": percentile(ttft_values, 95),
            "p99": percentile(ttft_values, 99),
        },
        "e2e_ms": {
            "mean": mean(e2e_values),
            "median": median(e2e_values),
            "p90": percentile(e2e_values, 90),
            "p95": percentile(e2e_values, 95),
            "p99": percentile(e2e_values, 99),
        },
        "tpot_ms": {
            "mean": mean(tpot_values),
            "p99": percentile(tpot_values, 99),
        },
        "rtf": {
            "mean": mean(rtf_values),
            "p99": percentile(rtf_values, 99),
        },
        "segments": {
            "mean": mean(segment_values),
            "p99": percentile(segment_values, 99),
        },
        "vad_time_ms": {
            "mean": mean(vad_time_values),
            "p99": percentile(vad_time_values, 99),
        },
    }


def perform_warmup(
    *,
    url: str,
    args: argparse.Namespace,
) -> list[float]:
    if args.no_warmup or not args.warmup_file:
        return []

    warmup_path = Path(args.warmup_file)
    if not warmup_path.exists():
        print(f"WARNING: warmup file not found: {args.warmup_file}")
        return []

    print(f"Running {args.warmup_iterations} warmup request(s)...")
    times: list[float] = []
    duration_s = get_audio_duration_seconds(warmup_path)
    event = threading.Event()
    event.set()
    for index in range(args.warmup_iterations):
        result = run_one_request(
            index + 1,
            warmup_path,
            duration_s,
            url,
            args,
            event,
        )
        if result["ok"]:
            times.append(float(result["e2e_ms"]))
            print(
                f"  warmup {index + 1}: "
                f"ttft={result['ttft_ms']:.2f}ms, "
                f"e2e={result['e2e_ms']:.2f}ms"
            )
        else:
            print(f"  warmup {index + 1} failed: {result['error']}")
    return times


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    audio_paths = find_all_audio(args.audio_dir)
    if args.warmup_file:
        warmup_path = Path(args.warmup_file).resolve()
        audio_paths = [
            path for path in audio_paths if path.resolve() != warmup_path
        ]
    if not audio_paths:
        raise RuntimeError("no benchmark audio files remain")

    num_requests = args.num_requests or len(audio_paths)
    tasks: list[dict[str, Any]] = []
    if args.mode == "vad" and not args.live_vad:
        print("Precomputing VAD segments before benchmark...")

    for request_id in range(num_requests):
        audio_path = audio_paths[request_id % len(audio_paths)].resolve()
        task: dict[str, Any] = {
            "request_id": request_id + 1,
            "audio_path": audio_path,
            "audio_duration_s": get_audio_duration_seconds(audio_path),
        }
        if args.mode == "vad" and not args.live_vad:
            precompute_start = time.perf_counter()
            task["precomputed"] = precompute_vad_request(audio_path, args=args)
            elapsed_s = time.perf_counter() - precompute_start
            precomputed = task["precomputed"]
            print(
                f"  {request_id + 1}/{num_requests} {audio_path.name}: "
                f"{len(precomputed.segments)} segments, "
                f"precompute={elapsed_s * 1000:.3f}ms"
            )
        tasks.append(task)

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=str, default=".")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--mode", choices=["full", "vad"], default="vad")
    parser.add_argument("--max-completion-tokens", type=int, default=1536)
    parser.add_argument("--segment-max-completion-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--segment-concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--output-file", type=str, default="benchmark_results.json")
    parser.add_argument("--warmup-file", type=str, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=700)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300)
    parser.add_argument("--vad-max-segment-s", type=float, default=25.0)
    parser.add_argument("--vad-chunk-duration-ms", type=int, default=1000)
    parser.add_argument("--vad-onnx", action="store_true")
    parser.add_argument("--head-fixed-segment-s", type=float, default=0.0)
    parser.add_argument(
        "--live-vad",
        action="store_true",
        help=(
            "Run local audio decode and VAD inside benchmark worker threads. "
            "By default, VAD segments are precomputed before measurement."
        ),
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.segment_concurrency < 1:
        raise ValueError("--segment-concurrency must be >= 1")
    if args.num_requests is not None and args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")
    if args.head_fixed_segment_s < 0:
        raise ValueError("--head-fixed-segment-s must be non-negative")

    url = f"http://{args.host}:{args.port}/v1/audio/transcriptions"
    warmup_times = perform_warmup(url=url, args=args)
    tasks = build_tasks(args)

    print(f"Benchmark mode: {args.mode}")
    print(f"Endpoint: {url}")
    print(f"Model: {args.model}")
    print(f"Benchmark requests: {len(tasks)}")
    print(f"Maximum request concurrency: {args.concurrency}")
    print(f"Segment concurrency per request: {args.segment_concurrency}")
    print(f"Head fixed segment: {args.head_fixed_segment_s:.3f}s")
    print(f"Precomputed VAD: {args.mode == 'vad' and not args.live_vad}")
    print("=" * 80)

    start_event = threading.Event()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_one_request,
                task["request_id"],
                task["audio_path"],
                task["audio_duration_s"],
                url,
                args,
                start_event,
                task.get("precomputed"),
            )
            for task in tasks
        ]

        benchmark_start = time.perf_counter()
        start_event.set()
        iterator = as_completed(futures)
        if tqdm is not None:
            progress = tqdm(total=len(futures), desc="Benchmark", unit="req")
            try:
                for future in iterator:
                    result = future.result()
                    results.append(result)
                    progress.set_postfix(
                        {
                            "success": sum(1 for item in results if item["ok"]),
                            "failed": sum(1 for item in results if not item["ok"]),
                            "file": result["file"],
                        }
                    )
                    progress.update(1)
            finally:
                progress.close()
        else:
            for index, future in enumerate(iterator, start=1):
                result = future.result()
                results.append(result)
                print(f"Progress: {index}/{len(futures)} requests completed")

        benchmark_duration_s = time.perf_counter() - benchmark_start

    summary = print_summary(
        results=results,
        concurrency=args.concurrency,
        benchmark_duration_s=benchmark_duration_s,
    )

    output = {
        "config": vars(args),
        "warmup": {
            "performed": bool(warmup_times),
            "iterations": len(warmup_times),
            "times_ms": warmup_times,
            "avg_ms": mean(warmup_times),
        },
        "summary": summary,
        "per_request": results,
    }
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
