# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python backend for the realtime ASR browser demo.

The browser only connects to this local backend. The backend serves the static
page and relays WebSocket messages between the page and the realtime model
server.
"""

import argparse
import asyncio
import functools
import inspect
import mimetypes
import pathlib
import threading
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websockets

DEFAULT_TARGET = "ws://135.30.72.115:1025/v1/realtime"


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


async def pipe(source, target, name: str) -> None:
    async for message in source:
        await target.send(message)
    print(f"[ws] {name} closed")


async def handle_browser_connection(browser_ws, default_target: str) -> None:
    request = getattr(browser_ws, "request", None)
    request_path = getattr(browser_ws, "path", None)
    if request_path is None and request is not None:
        request_path = getattr(request, "path", None)
    if request_path is None:
        request_path = "/ws"
    query = urllib.parse.urlparse(request_path).query
    params = urllib.parse.parse_qs(query)
    target_url = params.get("target", [default_target])[0] or default_target

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


async def start_ws_server(host: str, port: int, default_target: str) -> None:
    async def handler(websocket, _path=None):
        await handle_browser_connection(websocket, default_target)

    async with websockets.serve(handler, host, port):
        print(f"Browser WebSocket: ws://{host}:{port}/ws")
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
