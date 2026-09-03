# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json

from examples.speech_to_text.openai.analyze_qwen3_asr_streaming_probe import analyze


def _event(timestamp_ns, event, request_id, session_id="7", chunk_id=0, **extra):
    return {
        "schema_version": 1,
        "run_id": "test",
        "wall_time_ns": timestamp_ns,
        "monotonic_ns": timestamp_ns,
        "process": extra.pop("process", "api"),
        "event": event,
        "request_id": request_id,
        "external_request_id": request_id,
        "request_kind": "stream",
        "derived_session_id": session_id,
        "chunk_id": chunk_id,
        **extra,
    }


def test_analyzer_detects_vllm_overlap_and_qwen_queue(tmp_path):
    second = 1_000_000_000
    events = [
        _event(second, "request_received", "stream_7_0", chunk_id=0),
        _event(2 * second, "request_first_scheduled", "stream_7_0", chunk_id=0),
        _event(3 * second, "request_first_token", "stream_7_0", chunk_id=0),
        _event(5 * second, "request_received", "stream_7_1", chunk_id=1),
        _event(8 * second, "request_finished", "stream_7_0", chunk_id=0),
        _event(9 * second, "request_first_scheduled", "stream_7_1", chunk_id=1),
        _event(10 * second, "request_first_token", "stream_7_1", chunk_id=1),
        _event(12 * second, "request_finished", "stream_7_1", chunk_id=1),
    ]
    probe_path = tmp_path / "streaming_probe.api.1.jsonl"
    probe_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    (tmp_path / "qwen_server.log").write_text(
        "2026-09-03 12:00:00.100 - DEBUG - [conn-1] Audio append: "
        "duration=0.100s pending_bytes=3200 queue_size=2\n"
        "2026-09-03 12:00:01.100 - INFO - [conn-1] Commit received: "
        "queue_size=3 pending_bytes=6400 seg_audio_duration=2.000s\n"
        '2026-09-03 12:00:02.100 - INFO - [conn-1] Segment done: text="hi" '
        "lang=Chinese segment_latency=3.000s commit_latency=1.250s "
        "audio_duration=2.000s\n"
    )

    summary = analyze(tmp_path, warmup_seconds=0)

    assert summary["conclusion"] == "vllm_same_session_overlap"
    assert summary["overlapping_same_session_pairs"] == 1
    assert summary["qwen_queue_size"]["max"] == 3
    assert summary["qwen_commit_latency_ms"]["p95"] == 1250
    assert (tmp_path / "summary.json").exists()
    for name in ("requests.csv", "sessions.csv", "timeline.csv"):
        with (tmp_path / name).open(newline="") as source:
            assert list(csv.DictReader(source))


def test_analyzer_reports_qwen_only_backlog_and_missing_probe(tmp_path):
    (tmp_path / "qwen_server.log").write_text(
        "2026-09-03 12:00:00.100 - DEBUG - [conn-1] Audio append: "
        "duration=0.100s pending_bytes=3200 queue_size=4\n"
    )

    summary = analyze(tmp_path, warmup_seconds=0)

    assert summary["conclusion"] == "qwen_audio_queue_only_observed"
    assert summary["probe_event_count"] == 0
    assert summary["qwen_positive_queue_events"] == 1


def test_analyzer_accepts_untimed_qwen_log_and_explicit_path(tmp_path):
    qwen_log = tmp_path / "service-output.txt"
    qwen_log.write_text(
        "[ws-a] Audio append: duration=0.075s pending_bytes=156000 queue_size=11\n"
        "[ws-a] Commit received: queue_size=10 pending_bytes=156000 "
        "seg_audio_duration=4.875s\n"
        '[ws-a] Segment done: text="hi" lang=Chinese '
        "segment_latency=3.000s commit_latency=1.250s audio_duration=4.875s\n"
    )

    summary = analyze(tmp_path, warmup_seconds=120, qwen_log=qwen_log)

    assert summary["qwen_event_count"] == 3
    assert summary["qwen_untimed_event_count"] == 3
    assert summary["qwen_queue_size"]["max"] == 11
    assert summary["qwen_commit_latency_ms"]["p95"] == 1250
    assert summary["conclusion"] == "qwen_audio_queue_only_observed"


def test_single_just_enqueued_item_is_not_classified_as_backlog(tmp_path):
    (tmp_path / "qwen_server.log").write_text(
        "2026-09-03 12:00:00.100 - DEBUG - [conn-1] Audio append: "
        "duration=0.100s pending_bytes=3200 queue_size=1\n"
    )

    summary = analyze(tmp_path, warmup_seconds=0)

    assert summary["conclusion"] == "no_backlog_observed"
    assert summary["qwen_positive_queue_events"] == 1
    assert summary["qwen_backlog_queue_events"] == 0
