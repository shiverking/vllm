# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python backend for the realtime ASR browser demo.

The browser only connects to this local backend. The backend serves the static
page and relays WebSocket messages between the page and the realtime model
server.
"""

import argparse
import asyncio
import base64
import functools
import inspect
import json
import pathlib
import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websockets

DEFAULT_TARGET = "ws://135.30.72.115:1025/v1/realtime"
SAMPLE_RATE = 16_000
CHUNK_DURATION_MS = 100


class StaticFileHandler(SimpleHTTPRequestHandler):
    """Serve the web demo files without caching them."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[http] {self.address_string()} - {fmt % args}")


def start_http_server(host: str, port: int, directory: pathlib.Path) -> None:
    handler = functools.partial(StaticFileHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Page: http://{host}:{port}")
    server.serve_forever()


def websocket_connect(url: str):
    kwargs: dict[str, object] = {
        "open_timeout": 10,
        "close_timeout": 5,
    }
    signature = inspect.signature(websockets.connect)
    if "extra_headers" in signature.parameters:
        kwargs["extra_headers"] = {}
    elif "additional_headers" in signature.parameters:
        kwargs["additional_headers"] = {}
    return websockets.connect(url, **kwargs)


def get_request_path(websocket) -> str:
    request = getattr(websocket, "request", None)
    request_path = getattr(websocket, "path", None)
    if request_path is None and request is not None:
        request_path = getattr(request, "path", None)
    return request_path or "/ws"


def parse_query_value(request_path: str, key: str, default: str) -> str:
    query = urllib.parse.urlparse(request_path).query
    params = urllib.parse.parse_qs(query)
    return params.get(key, [default])[0] or default


async def pipe(source, target, name: str) -> None:
    async for message in source:
        await target.send(message)
    print(f"[ws] {name} closed")


async def handle_browser_connection(browser_ws, default_target: str) -> None:
    target_url = parse_query_value(get_request_path(browser_ws), "target",
                                   default_target)

    peer = getattr(browser_ws, "remote_address", "browser")
    print(f"[ws] Browser connected: {peer}")
    print(f"[ws] Relaying to: {target_url}")

    try:
        async with websocket_connect(target_url) as target_ws:
            tasks = [
                asyncio.create_task(pipe(browser_ws, target_ws, "browser -> model")),
                asyncio.create_task(pipe(target_ws, browser_ws, "model -> browser")),
            ]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except Exception as exc:
        print(f"[ws] Relay failed: {exc!r}")
        await browser_ws.close(code=1011, reason="backend relay failed")


def choose_loopback_microphone(soundcard):
    default_speaker = soundcard.default_speaker()
    microphones = soundcard.all_microphones(include_loopback=True)

    for microphone in microphones:
        if microphone.name == default_speaker.name:
            return microphone

    for microphone in microphones:
        if default_speaker.name in microphone.name:
            return microphone

    for microphone in microphones:
        name = microphone.name.lower()
        if "loopback" in name or getattr(microphone, "isloopback", False):
            return microphone

    raise RuntimeError("No loopback recording device was found.")


def frames_to_pcm16_base64(frames) -> str:
    import numpy as np

    audio = np.asarray(frames, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype("<i2")
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


async def capture_system_audio(target_ws, browser_ws, stop_event: asyncio.Event):
    try:
        import soundcard
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install soundcard and numpy to capture "
            "system audio."
        ) from exc

    microphone = choose_loopback_microphone(soundcard)
    chunk_frames = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

    await browser_ws.send(
        json.dumps({
            "type": "system_audio.started",
            "device": microphone.name,
        })
    )
    print(f"[audio] Capturing system audio from: {microphone.name}")

    with microphone.recorder(samplerate=SAMPLE_RATE, channels=2) as recorder:
        while not stop_event.is_set():
            frames = await asyncio.to_thread(
                recorder.record,
                numframes=chunk_frames,
            )
            await target_ws.send(
                json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": frames_to_pcm16_base64(frames),
                })
            )


async def read_system_audio_controls(browser_ws, stop_event: asyncio.Event):
    async for message in browser_ws:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "system_audio.stop":
            stop_event.set()
            return

        if event.get("type") == "input_audio_buffer.commit" and event.get(
                "final"):
            stop_event.set()
            return


async def forward_model_messages_until_done(source, target) -> None:
    async for message in source:
        await target.send(message)
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            continue

        if event.get("type") in ("transcription.done", "error"):
            return


async def send_final_commit(target_ws) -> None:
    try:
        await target_ws.send(
            json.dumps({
                "type": "input_audio_buffer.commit",
                "final": True,
            }))
    except Exception as exc:
        print(f"[ws] Failed to send final commit: {exc!r}")


async def handle_system_audio_connection(browser_ws, default_target: str) -> None:
    request_path = get_request_path(browser_ws)
    target_url = parse_query_value(request_path, "target", default_target)
    model = parse_query_value(request_path, "model", "")

    peer = getattr(browser_ws, "remote_address", "browser")
    print(f"[system-audio] Browser connected: {peer}")
    print(f"[system-audio] Relaying to: {target_url}")

    stop_event = asyncio.Event()
    try:
        async with websocket_connect(target_url) as target_ws:
            if model:
                await target_ws.send(
                    json.dumps({
                        "type": "session.update",
                        "model": model,
                    }))
            await target_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            capture_task = asyncio.create_task(
                capture_system_audio(target_ws, browser_ws, stop_event))
            model_task = asyncio.create_task(
                forward_model_messages_until_done(target_ws, browser_ws))
            control_task = asyncio.create_task(
                read_system_audio_controls(browser_ws, stop_event))

            done, _pending = await asyncio.wait(
                [capture_task, model_task, control_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if control_task in done:
                control_task.result()
                stop_event.set()
                await capture_task
                await send_final_commit(target_ws)
                await model_task
            elif model_task in done:
                model_task.result()
                stop_event.set()
                await capture_task
            else:
                capture_task.result()

            for task in (capture_task, model_task, control_task):
                if not task.done():
                    task.cancel()
    except Exception as exc:
        print(f"[system-audio] Failed: {exc!r}")
        await browser_ws.send(
            json.dumps({
                "type": "error",
                "error": {
                    "message": str(exc),
                },
            }))
        await browser_ws.close(code=1011, reason="system audio failed")


async def start_ws_server(host: str, port: int, default_target: str) -> None:
    async def handler(websocket, _path=None):
        request_path = get_request_path(websocket)
        if urllib.parse.urlparse(request_path).path == "/system-audio":
            await handle_system_audio_connection(websocket, default_target)
        else:
            await handle_browser_connection(websocket, default_target)

    async with websockets.serve(handler, host, port):
        print(f"Browser WebSocket: ws://{host}:{port}/ws")
        print(f"System audio WS:   ws://{host}:{port}/system-audio")
        print(f"Default target:    {default_target}")
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime ASR web demo backend")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Default realtime model WebSocket URL.",
    )
    parser.add_argument(
        "--http-host",
        default="127.0.0.1",
        help="Host for the static web page.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Port for the static web page.",
    )
    parser.add_argument(
        "--ws-host",
        default="127.0.0.1",
        help="Host for the browser WebSocket endpoint.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="Port for the browser WebSocket endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = pathlib.Path(__file__).resolve().parent

    http_thread = threading.Thread(
        target=start_http_server,
        args=(args.http_host, args.http_port, directory),
        daemon=True,
    )
    http_thread.start()

    asyncio.run(start_ws_server(args.ws_host, args.ws_port, args.target))


if __name__ == "__main__":
    main()
