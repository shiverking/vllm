# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.multimodal.realtime_vad import RealtimeVADSegmenter


class FakeSpeechDetector:

    def __init__(self, decisions: list[bool], frame_size: int = 4) -> None:
        self._decisions = decisions
        self._idx = 0
        self._frame_size = frame_size

    @property
    def frame_size(self) -> int:
        return self._frame_size

    def is_speech(self, audio: np.ndarray) -> bool:
        assert audio.shape[-1] == self._frame_size
        decision = self._decisions[self._idx]
        self._idx += 1
        return decision


def _frames(num_frames: int, frame_size: int = 4) -> np.ndarray:
    return np.ones(num_frames * frame_size, dtype=np.float32)


def test_vad_segmenter_emits_after_silence() -> None:
    detector = FakeSpeechDetector([False, True, True, False, False])
    segmenter = RealtimeVADSegmenter(
        detector,
        sampling_rate=1000,
        min_speech_duration_ms=8,
        min_silence_duration_ms=8,
        speech_pad_ms=4,
        max_segment_duration_s=1.0,
    )

    segments = segmenter.write_audio(_frames(5))

    assert len(segments) == 1
    # One pre-roll frame, two speech frames, and one trailing pad frame.
    assert segments[0].shape[-1] == 16


def test_vad_segmenter_drops_too_short_speech() -> None:
    detector = FakeSpeechDetector([True, False, False])
    segmenter = RealtimeVADSegmenter(
        detector,
        sampling_rate=1000,
        min_speech_duration_ms=8,
        min_silence_duration_ms=8,
        speech_pad_ms=4,
        max_segment_duration_s=1.0,
    )

    assert segmenter.write_audio(_frames(3)) == []
    assert segmenter.flush() is None


def test_vad_segmenter_forces_max_segment_duration() -> None:
    detector = FakeSpeechDetector([True, True, True, True])
    segmenter = RealtimeVADSegmenter(
        detector,
        sampling_rate=1000,
        min_speech_duration_ms=4,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        max_segment_duration_s=0.012,
    )

    segments = segmenter.write_audio(_frames(4))

    assert len(segments) == 1
    assert segments[0].shape[-1] == 12


def test_vad_segmenter_flushes_remaining_speech() -> None:
    detector = FakeSpeechDetector([True, True])
    segmenter = RealtimeVADSegmenter(
        detector,
        sampling_rate=1000,
        min_speech_duration_ms=8,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        max_segment_duration_s=1.0,
    )

    assert segmenter.write_audio(_frames(2)) == []
    remaining = segmenter.flush()

    assert remaining is not None
    assert remaining.shape[-1] == 8
