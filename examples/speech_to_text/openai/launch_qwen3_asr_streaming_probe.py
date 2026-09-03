# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Launch Qwen3-ASR with vLLM probes and timestamped service logs."""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

_FLUSH_INTERVAL_S = 0.1
_FLUSH_BATCH_SIZE = 256


def _capture_line(line: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"{timestamp} - CAPTURE - {line.rstrip()}\n"


def run(log_dir: Path, command: list[str], tee: bool = False) -> int:
    if not command:
        raise ValueError("A service command is required after '--'")

    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "qwen_server.log"
    if log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing log: {log_path}. Use a new run directory."
        )

    child_env = os.environ.copy()
    child_env["VLLM_STREAMING_PROBE_LOG_DIR"] = str(log_dir)

    with log_path.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        try:
            _copy_output(process.stdout, output, tee)
            return process.wait()
        except KeyboardInterrupt:
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                return process.wait()


def _copy_output(source: TextIO, output: TextIO, tee: bool) -> None:
    last_flush = time.monotonic()
    pending_lines = 0
    for line in source:
        captured = _capture_line(line)
        output.write(captured)
        pending_lines += 1
        if tee:
            sys.stdout.write(captured)
        now = time.monotonic()
        if (
            pending_lines >= _FLUSH_BATCH_SIZE
            or now - last_flush >= _FLUSH_INTERVAL_S
        ):
            output.flush()
            if tee:
                sys.stdout.flush()
            pending_lines = 0
            last_flush = now
    output.flush()
    if tee:
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument(
        "--tee",
        action="store_true",
        help="Also print timestamped service output to the terminal",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return_code = run(args.log_dir, command, args.tee)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
