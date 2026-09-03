# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
import sys

import pytest

from examples.speech_to_text.openai.launch_qwen3_asr_streaming_probe import run


def test_launcher_sets_probe_dir_and_timestamps_output(tmp_path):
    command = [
        sys.executable,
        "-c",
        "import os; print('[ws-a] Audio append: queue_size=2'); "
        "print(os.environ['VLLM_STREAMING_PROBE_LOG_DIR'])",
    ]

    assert run(tmp_path, command) == 0

    lines = (tmp_path / "qwen_server.log").read_text().splitlines()
    assert re.match(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} - CAPTURE - ",
        lines[0],
    )
    assert lines[0].endswith("[ws-a] Audio append: queue_size=2")
    assert lines[1].endswith(str(tmp_path.resolve()))


def test_launcher_refuses_to_overwrite_existing_log(tmp_path):
    (tmp_path / "qwen_server.log").write_text("existing\n")

    with pytest.raises(FileExistsError):
        run(tmp_path, [sys.executable, "-c", "print('new')"])
