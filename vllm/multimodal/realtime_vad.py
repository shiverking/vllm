# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from functools import cache
from typing import Protocol

import numpy as np


class SpeechDetector(Protocol):

    @property
    def frame_size(self) -> int: ...

    def is_speech(self, audio: np.ndarray) -> bool: ...


@cache
def _load_silero_vad_model():
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise ImportError(
            "Silero VAD realtime segmentation requires the optional "
            "`silero-vad` package. Install it to use "
            "VLLM_QWEN3_ASR_REALTIME_VAD_BACKEND=silero."
        ) from exc

    return load_silero_vad()


class SileroSpeechDetector:

    def __init__(
        self,
        *,
        sampling_rate: int,
        threshold: float,
    ) -> None:
        if sampling_rate not in (8000, 16000):
            raise ValueError(
                "Silero VAD supports 8000 Hz and 16000 Hz audio, "
                f"got {sampling_rate} Hz."
            )
        self._sampling_rate = sampling_rate
        self._threshold = threshold
        self._frame_size = 256 if sampling_rate == 8000 else 512
        self._model = _load_silero_vad_model()

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
    """Turn streaming audio into utterance-like segments using VAD decisions."""

    def __init__(
        self,
        detector: SpeechDetector,
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
            frame = self._pending_audio[: self._frame_size].copy()
            self._pending_audio = self._pending_audio[self._frame_size :]

            if self._detector.is_speech(frame):
                self._append_speech_frame(frame)
            elif self._in_speech:
                self._append_silence_frame(frame)
            else:
                self._append_pre_roll(frame)

            if self._in_speech and self._silence_samples >= self._min_silence_samples:
                segment = self._flush_active_segment(trim_trailing_silence=True)
                if segment is not None:
                    segments.append(segment)
            elif (
                self._in_speech
                and self._segment_samples >= self._max_segment_samples
            ):
                segment = self._flush_active_segment(trim_trailing_silence=False)
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
            self._pre_roll_audio = self._pre_roll_audio[-self._speech_pad_samples :]

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
