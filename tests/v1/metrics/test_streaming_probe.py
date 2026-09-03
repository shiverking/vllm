# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import threading

import pytest

import vllm.v1.metrics.streaming_probe as probe


def test_classify_qwen_request_ids():
    assert probe.classify_request_id("stream_123_4") == ("stream", "123", 4)
    assert probe.classify_request_id("stream_123_4-deadbeef") == (
        "stream",
        "123",
        4,
    )
    assert probe.classify_request_id("finish_123-deadbeef") == (
        "finish",
        "123",
        None,
    )
    assert probe.classify_request_id("unrelated") == ("other", None, None)


def test_probe_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_STREAMING_PROBE_LOG_DIR", raising=False)
    probe.close_streaming_probe_writers()
    assert probe.get_streaming_probe_writer("api") is None
    probe.emit_streaming_probe("api", "request_received", "request")


def test_writer_emits_jsonl_and_deduplicates_once_events(tmp_path):
    writer = probe.StreamingProbeWriter(str(tmp_path), "api")
    writer.emit(
        "request_received",
        "stream_42_3-deadbeef",
        external_request_id="stream_42_3",
        prompt_tokens=12,
    )
    writer.emit("request_received", "stream_42_3-deadbeef")
    writer.emit("streaming_update_queued", "stream_42_3-deadbeef")
    writer.emit("streaming_update_queued", "stream_42_3-deadbeef")
    writer.close()

    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "request_received",
        "streaming_update_queued",
        "streaming_update_queued",
    ]
    assert records[0]["request_kind"] == "stream"
    assert records[0]["derived_session_id"] == "42"
    assert records[0]["chunk_id"] == 3
    assert records[0]["prompt_tokens"] == 12


def test_writer_accepts_concurrent_emitters(tmp_path):
    writer = probe.StreamingProbeWriter(str(tmp_path), "engine")

    def emit_batch(worker: int):
        for index in range(50):
            writer.emit("streaming_update_queued", f"request-{worker}-{index}")

    threads = [
        threading.Thread(target=emit_batch, args=(worker,)) for worker in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.close()

    records = writer.path.read_text().splitlines()
    assert len(records) == 200
    assert all(json.loads(line)["process"] == "engine" for line in records)


def test_writer_rotates(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(probe, "_MAX_FILE_BYTES", 100)
    writer = probe.StreamingProbeWriter(str(tmp_path), "api")
    writer._write_batch([{"payload": "x" * 80}])
    writer._write_batch([{"payload": "y" * 80}])
    writer.close()

    assert writer.path.exists()
    assert writer.path.with_name(f"{writer.path.name}.1").exists()


def test_writer_failure_does_not_escape_emit(tmp_path, monkeypatch, caplog):
    writer = probe.StreamingProbeWriter(str(tmp_path), "api")

    def fail(_records):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_batch", fail)
    writer.emit("streaming_update_queued", "request")
    writer.close()
    assert "disabled after error" in caplog.text
