# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead, opt-in lifecycle tracing for streaming requests."""

import atexit
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

from vllm import envs

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_FLUSH_INTERVAL_S = 0.1
_FLUSH_BATCH_SIZE = 256
_MAX_FILE_BYTES = 256 * 1024 * 1024
_BACKUP_COUNT = 3
_STOP = object()
_REQUEST_PATTERNS = (
    ("stream", re.compile(r"^stream_(?P<session>\d+)_(?P<chunk>\d+)(?:-.+)?$")),
    ("finish", re.compile(r"^finish_(?P<session>\d+)(?:-.+)?$")),
)
_ONCE_EVENTS = frozenset(
    {
        "request_received",
        "engine_request_added",
        "request_queued",
        "request_first_scheduled",
        "request_first_token",
        "request_finished",
    }
)


def classify_request_id(request_id: str) -> tuple[str, str | None, int | None]:
    """Classify the Qwen ASR diagnostic request naming convention."""
    for request_kind, pattern in _REQUEST_PATTERNS:
        if match := pattern.match(request_id):
            chunk = match.groupdict().get("chunk")
            return request_kind, match.group("session"), int(chunk) if chunk else None
    return "other", None, None


class StreamingProbeWriter:
    """Write probe records from a background thread into a per-process JSONL."""

    def __init__(self, log_dir: str, process: str):
        self.process = process
        self.pid = os.getpid()
        self.log_dir = Path(log_dir)
        self.path = self.log_dir / f"streaming_probe.{process}.{self.pid}.jsonl"
        self.run_id = self.log_dir.name
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue()
        self._seen: set[tuple[str, str]] = set()
        self._seen_lock = threading.Lock()
        self._closed = False
        self._warned = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"vllm-streaming-probe-{process}",
            daemon=True,
        )
        self._thread.start()

    def emit(self, event: str, request_id: str, **fields: Any) -> None:
        if self._closed:
            return
        if event in _ONCE_EVENTS:
            key = (event, request_id)
            with self._seen_lock:
                if key in self._seen:
                    return
                self._seen.add(key)

        external_request_id = fields.pop("external_request_id", None)
        parsed_id = external_request_id or request_id
        request_kind, session_id, chunk_id = classify_request_id(parsed_id)
        record = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": self.run_id,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "process": self.process,
            "event": event,
            "request_id": request_id,
            "external_request_id": external_request_id,
            "request_kind": request_kind,
            "derived_session_id": session_id,
            "chunk_id": chunk_id,
            "status": fields.pop("status", None),
            "num_running": fields.pop("num_running", None),
            "num_waiting": fields.pop("num_waiting", None),
            "prompt_tokens": fields.pop("prompt_tokens", None),
            "cached_tokens": fields.pop("cached_tokens", None),
            "output_tokens": fields.pop("output_tokens", None),
            "finish_reason": fields.pop("finish_reason", None),
        }
        record.update(fields)
        self._queue.put(record)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=5.0)

    def _warn_once(self, error: BaseException) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("Streaming probe writer disabled after error: %s", error)

    def _rotate(self, incoming_bytes: int) -> None:
        if (
            not self.path.exists()
            or self.path.stat().st_size + incoming_bytes <= _MAX_FILE_BYTES
        ):
            return
        oldest = self.path.with_name(f"{self.path.name}.{_BACKUP_COUNT}")
        oldest.unlink(missing_ok=True)
        for index in range(_BACKUP_COUNT - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def _write_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        payload = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._rotate(len(payload.encode("utf-8")))
        with self.path.open("a", encoding="utf-8") as output:
            output.write(payload)
            output.flush()

    def _run(self) -> None:
        stopping = False
        while not stopping:
            batch: list[dict[str, Any]] = []
            try:
                item = self._queue.get(timeout=_FLUSH_INTERVAL_S)
                if item is _STOP:
                    stopping = True
                else:
                    batch.append(item)  # type: ignore[arg-type]
                deadline = time.monotonic() + _FLUSH_INTERVAL_S
                while len(batch) < _FLUSH_BATCH_SIZE:
                    try:
                        item = self._queue.get(
                            timeout=max(0.0, deadline - time.monotonic())
                        )
                    except queue.Empty:
                        break
                    if item is _STOP:
                        stopping = True
                        break
                    batch.append(item)  # type: ignore[arg-type]
            except queue.Empty:
                pass

            if batch:
                try:
                    self._write_batch(batch)
                except Exception as error:
                    self._warn_once(error)
                    return


_writers: dict[str, StreamingProbeWriter] = {}
_writers_lock = threading.Lock()


def get_streaming_probe_writer(process: str) -> StreamingProbeWriter | None:
    log_dir = envs.VLLM_STREAMING_PROBE_LOG_DIR
    if not log_dir:
        return None
    pid = os.getpid()
    with _writers_lock:
        writer = _writers.get(process)
        if writer is None or writer.pid != pid:
            writer = StreamingProbeWriter(log_dir, process)
            _writers[process] = writer
        return writer


def emit_streaming_probe(
    process: str, event: str, request_id: str, **fields: Any
) -> None:
    if writer := get_streaming_probe_writer(process):
        writer.emit(event, request_id, **fields)


def close_streaming_probe_writers() -> None:
    with _writers_lock:
        writers = list(_writers.values())
        _writers.clear()
    for writer in writers:
        writer.close()


atexit.register(close_streaming_probe_writers)
