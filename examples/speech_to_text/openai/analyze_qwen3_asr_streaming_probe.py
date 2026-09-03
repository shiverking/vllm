# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Analyze Qwen3-ASR service logs and vLLM streaming probe JSONL files."""

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

_QWEN_TIMESTAMP = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
_CONNECTION = r"\[(?P<connection>[^\]]+)\]"
_QWEN_PATTERNS = (
    (
        "audio_append",
        re.compile(_CONNECTION + r" Audio append:.*queue_size=(?P<queue_size>\d+)"),
    ),
    (
        "commit_received",
        re.compile(_CONNECTION + r" Commit received: queue_size=(?P<queue_size>\d+)"),
    ),
    (
        "chunk_first_token",
        re.compile(
            _CONNECTION
            + r" chunk_id=(?P<chunk_id>\d+) prefill_ms=(?P<prefill_ms>[0-9.]+)"
        ),
    ),
    (
        "chunk_finished",
        re.compile(
            _CONNECTION
            + r" chunk_id=(?P<chunk_id>\d+).*decode_total_ms=(?P<decode_ms>[0-9.]+)"
        ),
    ),
    (
        "segment_done",
        re.compile(
            _CONNECTION
            + r" Segment done:.*segment_latency=(?P<segment_latency>[0-9.]+)s "
            + r"commit_latency=(?P<commit_latency>[0-9.]+)s "
            + r"audio_duration=(?P<audio_duration>[0-9.]+)s"
        ),
    ),
)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as output:
        if not fieldnames:
            return
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_probe_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("streaming_probe.*.jsonl*")):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}"
                    ) from error
                record["source_file"] = path.name
                events.append(record)
    return events


def load_qwen_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            timestamp_match = _QWEN_TIMESTAMP.search(line)
            for event_name, pattern in _QWEN_PATTERNS:
                if match := pattern.search(line):
                    wall_time_ns = None
                    if timestamp_match is not None:
                        wall_time_ns = int(
                            datetime.strptime(
                                timestamp_match.group("ts"),
                                "%Y-%m-%d %H:%M:%S.%f",
                            ).timestamp()
                            * 1_000_000_000
                        )
                    record: dict[str, Any] = {
                        "schema_version": 1,
                        "wall_time_ns": wall_time_ns,
                        "timestamp_source": (
                            "capture"
                            if timestamp_match is not None
                            and " - CAPTURE - " in line
                            else "log"
                            if timestamp_match is not None
                            else "unavailable"
                        ),
                        "process": "qwen",
                        "event": event_name,
                        "connection_id": match.group("connection"),
                        "source_file": path.name,
                        "source_line": line_number,
                    }
                    for key, value in match.groupdict().items():
                        if key == "connection" or value is None:
                            continue
                        record[key] = float(value) if "." in value else int(value)
                    events.append(record)
                    break
    return events


def _event_time_ms(record: dict[str, Any]) -> float:
    return record["wall_time_ns"] / 1_000_000.0


def _sort_key(record: dict[str, Any]) -> tuple[bool, int]:
    wall_time_ns = record.get("wall_time_ns")
    return (
        wall_time_ns is None,
        wall_time_ns if wall_time_ns is not None else record.get("source_line", 0),
    )


def analyze(
    run_dir: Path,
    warmup_seconds: float = 120.0,
    qwen_log: Path | None = None,
) -> dict[str, Any]:
    probe_events = load_probe_events(run_dir)
    qwen_log = qwen_log or run_dir / "qwen_server.log"
    qwen_events = load_qwen_events(qwen_log)
    all_events = sorted(probe_events + qwen_events, key=_sort_key)
    timed_events = [
        event for event in all_events if event.get("wall_time_ns") is not None
    ]
    if timed_events:
        cutoff_ns = timed_events[0]["wall_time_ns"] + int(warmup_seconds * 1e9)
        request_start_ns: dict[str, int] = {}
        for event in probe_events:
            if event.get("event") in {"request_received", "engine_request_added"}:
                request_start_ns.setdefault(event["request_id"], event["wall_time_ns"])
        analysis_events = sorted(
            [
                event
                for event in probe_events
                if request_start_ns.get(event.get("request_id"), event["wall_time_ns"])
                >= cutoff_ns
            ]
            + [
                event
                for event in qwen_events
                if event["wall_time_ns"] is None
                or event["wall_time_ns"] >= cutoff_ns
            ],
            key=_sort_key,
        )
    else:
        cutoff_ns = None
        # Without timestamps, warmup filtering and cross-layer correlation are
        # impossible. Keep the events for queue and embedded-latency statistics.
        analysis_events = list(qwen_events)

    request_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in analysis_events:
        if event.get("request_id"):
            request_events[event["request_id"]].append(event)

    request_rows: list[dict[str, Any]] = []
    for request_id, events in request_events.items():
        by_name = {event["event"]: event for event in events}
        received = by_name.get("request_received") or by_name.get(
            "engine_request_added"
        )
        scheduled = by_name.get("request_first_scheduled")
        first_token = by_name.get("request_first_token")
        finished = by_name.get("request_finished")
        representative = received or events[0]

        def delta_ms(end: dict[str, Any] | None, start: dict[str, Any] | None):
            if end is None or start is None:
                return None
            return _event_time_ms(end) - _event_time_ms(start)

        request_rows.append(
            {
                "request_id": request_id,
                "external_request_id": representative.get("external_request_id"),
                "request_kind": representative.get("request_kind"),
                "derived_session_id": representative.get("derived_session_id"),
                "chunk_id": representative.get("chunk_id"),
                "received_ns": received.get("wall_time_ns") if received else None,
                "scheduled_ns": scheduled.get("wall_time_ns") if scheduled else None,
                "first_token_ns": (
                    first_token.get("wall_time_ns") if first_token else None
                ),
                "finished_ns": finished.get("wall_time_ns") if finished else None,
                "queue_wait_ms": delta_ms(scheduled, received),
                "ttft_ms": delta_ms(first_token, received),
                "prefill_ms": delta_ms(first_token, scheduled),
                "decode_ms": delta_ms(finished, first_token),
                "prompt_tokens": representative.get("prompt_tokens"),
                "cached_tokens": (
                    first_token.get("cached_tokens") if first_token else None
                ),
                "output_tokens": finished.get("output_tokens") if finished else None,
                "finish_reason": finished.get("finish_reason") if finished else None,
            }
        )

    vllm_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        if row["derived_session_id"] is not None:
            vllm_sessions[str(row["derived_session_id"])].append(row)

    session_rows: list[dict[str, Any]] = []
    overlap_pairs = 0
    comparable_pairs = 0
    supersedable_output_tokens_upper_bound = 0
    for session_id, requests in vllm_sessions.items():
        stream_requests = sorted(
            (row for row in requests if row["request_kind"] == "stream"),
            key=lambda row: row["chunk_id"],
        )
        session_overlaps = 0
        session_pairs = 0
        for previous, current in zip(stream_requests, stream_requests[1:]):
            if previous["finished_ns"] is None or current["received_ns"] is None:
                continue
            session_pairs += 1
            if current["received_ns"] < previous["finished_ns"]:
                session_overlaps += 1
                supersedable_output_tokens_upper_bound += previous["output_tokens"] or 0
        comparable_pairs += session_pairs
        overlap_pairs += session_overlaps
        session_rows.append(
            {
                "source": "vllm",
                "session_id": session_id,
                "request_count": len(requests),
                "stream_request_count": len(stream_requests),
                "comparable_pairs": session_pairs,
                "overlap_pairs": session_overlaps,
                "has_overlap": session_overlaps > 0,
                "max_audio_queue": None,
            }
        )

    qwen_by_connection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in analysis_events:
        if event.get("process") == "qwen":
            qwen_by_connection[event["connection_id"]].append(event)
    positive_queue_events = 0
    backlog_queue_events = 0
    queue_values: list[float] = []
    qwen_commit_latencies_ms: list[float] = []
    qwen_segment_latencies_ms: list[float] = []
    for connection_id, events in qwen_by_connection.items():
        values = [
            float(event["queue_size"])
            for event in events
            if event["event"] in {"audio_append", "commit_received"}
            and event.get("queue_size") is not None
        ]
        positive_queue_events += sum(value > 0 for value in values)
        # A queue size of one is the item just enqueued and can be consumed
        # immediately. More than one proves that an earlier item is pending.
        backlog_queue_events += sum(value > 1 for value in values)
        queue_values.extend(values)
        commit_latencies_ms = [
            float(event["commit_latency"]) * 1000.0
            for event in events
            if event["event"] == "segment_done"
        ]
        segment_latencies_ms = [
            float(event["segment_latency"]) * 1000.0
            for event in events
            if event["event"] == "segment_done"
        ]
        qwen_commit_latencies_ms.extend(commit_latencies_ms)
        qwen_segment_latencies_ms.extend(segment_latencies_ms)
        session_rows.append(
            {
                "source": "qwen",
                "session_id": connection_id,
                "request_count": None,
                "stream_request_count": None,
                "comparable_pairs": None,
                "overlap_pairs": None,
                "has_overlap": None,
                "max_audio_queue": max(values) if values else None,
                "commit_latency_p95_ms": percentile(commit_latencies_ms, 95),
                "segment_latency_p95_ms": percentile(segment_latencies_ms, 95),
            }
        )

    queue_waits = [
        row["queue_wait_ms"] for row in request_rows if row["queue_wait_ms"] is not None
    ]
    ttfts = [row["ttft_ms"] for row in request_rows if row["ttft_ms"] is not None]
    final_rows = [row for row in request_rows if row["request_kind"] == "finish"]
    final_ttfts = [row["ttft_ms"] for row in final_rows if row["ttft_ms"] is not None]
    final_done = [
        (row["finished_ns"] - row["received_ns"]) / 1_000_000.0
        for row in final_rows
        if row["finished_ns"] is not None and row["received_ns"] is not None
    ]
    if overlap_pairs:
        conclusion = "vllm_same_session_overlap"
    elif backlog_queue_events:
        conclusion = "qwen_audio_queue_only_observed"
    else:
        conclusion = "no_backlog_observed"

    max_streaming_queue_depth = max(
        (
            int(event.get("streaming_queue_depth", 0))
            for event in analysis_events
            if event.get("event") == "streaming_update_queued"
        ),
        default=0,
    )
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "warmup_seconds": warmup_seconds,
        "cutoff_ns": cutoff_ns,
        "probe_event_count": len(probe_events),
        "qwen_event_count": len(qwen_events),
        "qwen_untimed_event_count": sum(
            event["wall_time_ns"] is None for event in qwen_events
        ),
        "qwen_log": str(qwen_log),
        "analyzed_event_count": len(analysis_events),
        "request_count": len(request_rows),
        "vllm_session_count": len(vllm_sessions),
        "comparable_same_session_pairs": comparable_pairs,
        "overlapping_same_session_pairs": overlap_pairs,
        "overlap_ratio": overlap_pairs / comparable_pairs if comparable_pairs else None,
        "supersedable_request_count": overlap_pairs,
        "supersedable_output_tokens_upper_bound": (
            supersedable_output_tokens_upper_bound
        ),
        "qwen_positive_queue_events": positive_queue_events,
        "qwen_backlog_queue_events": backlog_queue_events,
        "qwen_queue_size": {
            "p50": percentile(queue_values, 50),
            "p95": percentile(queue_values, 95),
            "p99": percentile(queue_values, 99),
            "max": max(queue_values) if queue_values else None,
        },
        "qwen_commit_latency_ms": {
            "p50": percentile(qwen_commit_latencies_ms, 50),
            "p95": percentile(qwen_commit_latencies_ms, 95),
            "p99": percentile(qwen_commit_latencies_ms, 99),
        },
        "qwen_segment_latency_ms": {
            "p50": percentile(qwen_segment_latencies_ms, 50),
            "p95": percentile(qwen_segment_latencies_ms, 95),
            "p99": percentile(qwen_segment_latencies_ms, 99),
        },
        "vllm_queue_wait_ms": {
            "p50": percentile(queue_waits, 50),
            "p95": percentile(queue_waits, 95),
            "p99": percentile(queue_waits, 99),
        },
        "vllm_ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
        },
        "final_request_ttft_ms": {
            "p50": percentile(final_ttfts, 50),
            "p95": percentile(final_ttfts, 95),
            "p99": percentile(final_ttfts, 99),
        },
        "final_request_done_ms": {
            "p50": percentile(final_done, 50),
            "p95": percentile(final_done, 95),
            "p99": percentile(final_done, 99),
        },
        "max_streaming_queue_depth": max_streaming_queue_depth,
        "conclusion": conclusion,
        "correlation_limit": (
            "Untimed Qwen events are included in queue and embedded-latency "
            "statistics, but cannot be warmup-filtered or correlated by time. "
            "Qwen connection_id and id(state) are also not mapped by the existing "
            "service; Qwen queue backlog and vLLM request overlap are reported "
            "independently."
        ),
    }

    timeline_rows = []
    for event in all_events:
        timeline_rows.append(
            {
                "wall_time_ns": event.get("wall_time_ns"),
                "timestamp_source": event.get("timestamp_source"),
                "process": event.get("process"),
                "event": event.get("event"),
                "request_id": event.get("request_id"),
                "derived_session_id": event.get("derived_session_id"),
                "connection_id": event.get("connection_id"),
                "chunk_id": event.get("chunk_id"),
                "status": event.get("status"),
                "num_running": event.get("num_running"),
                "num_waiting": event.get("num_waiting"),
                "queue_size": event.get("queue_size"),
                "streaming_queue_depth": event.get("streaming_queue_depth"),
                "commit_latency": event.get("commit_latency"),
                "segment_latency": event.get("segment_latency"),
            }
        )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(run_dir / "requests.csv", request_rows)
    _write_csv(run_dir / "sessions.csv", session_rows)
    _write_csv(run_dir / "timeline.csv", timeline_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--warmup-seconds", type=float, default=120.0)
    parser.add_argument(
        "--qwen-log",
        type=Path,
        help="Qwen service log path (default: <run_dir>/qwen_server.log)",
    )
    args = parser.parse_args()
    summary = analyze(args.run_dir, args.warmup_seconds, args.qwen_log)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
