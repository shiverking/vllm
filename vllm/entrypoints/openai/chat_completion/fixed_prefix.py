# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fixed-template image endpoint for the local capacity experiment."""

import asyncio
import copy
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import with_cancellation
from vllm.sampling_params import SamplingParams


def get_registry(raw_request):
    handler = raw_request.app.state.openai_serving_chat
    if not handler.engine_client.vllm_config.additional_config.get(
        "fixed_prefix_cache"
    ):
        raise HTTPException(404, "Fixed prefix experiment is disabled")
    if not hasattr(handler, "_fixed_prefix_templates"):
        handler._fixed_prefix_templates = {}
        handler._fixed_prefix_lock = asyncio.Lock()
    return handler, handler._fixed_prefix_templates


async def utility(handler, operation, *args):
    try:
        return await handler.engine_client.engine_core.call_utility_async(
            "fixed_prefix", operation, list(args)
        )
    except Exception as exc:
        raise HTTPException(409, str(exc)) from exc


async def register_prefix(raw_request: Request):
    handler, registry = get_registry(raw_request)
    body = await raw_request.json()
    request = ChatCompletionRequest.model_validate(body)
    if request.stream or request.n not in (None, 1):
        raise HTTPException(400, "Only non-streaming n=1 supported")
    async with handler._fixed_prefix_lock:
        rendered = await handler.render_chat_request(request)
        if isinstance(rendered, ErrorResponse):
            return JSONResponse(rendered.model_dump(), status_code=400)
        _, inputs = rendered
        if len(inputs) != 1 or inputs[0].get("type") != "multimodal":
            raise HTTPException(400, "Registration requires one image request")
        engine_input = inputs[0]
        positions = engine_input["mm_placeholders"]
        if set(positions) != {"image"} or len(positions["image"]) != 1:
            raise HTTPException(400, "Exactly one image is supported")
        boundary = positions["image"][0].offset
        tokens = engine_input["prompt_token_ids"][:boundary]
        prefix_id = uuid.uuid4().hex
        salt = request.cache_salt or f"fixed-prefix:{prefix_id}"
        await utility(handler, "prepare", prefix_id, tokens, salt)
        try:
            params = SamplingParams(
                max_tokens=1,
                temperature=0,
                extra_args={"fixed_prefix_capture": prefix_id},
            )
            async for _ in handler.engine_client.generate(
                {"prompt_token_ids": tokens, "cache_salt": salt},
                params,
                f"fixed-prefix-warm-{prefix_id}",
            ):
                pass
            status = await utility(handler, "status", prefix_id)
            if status["state"] != "ready":
                raise RuntimeError(f"Prefix checkpoint was not captured: {status}")
            template = copy.deepcopy(body)
            image_count = 0
            for message in template["messages"]:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part.get("type") == "image_url":
                            part["image_url"] = {"url": "<fixed-prefix-image>"}
                            part.pop("uuid", None)
                            image_count += 1
            if image_count != 1:
                raise ValueError("Exactly one image_url is supported")
            registry[prefix_id] = {
                "body": template,
                "salt": salt,
                "active": 0,
                "tokens": tokens,
            }
            return status
        except BaseException:
            await utility(handler, "unregister", prefix_id)
            raise


async def prefix_status(prefix_id: str, raw_request: Request):
    handler, registry = get_registry(raw_request)
    if prefix_id not in registry:
        raise HTTPException(404, "Unknown prefix_id")
    return await utility(handler, "status", prefix_id)


async def unregister_prefix(prefix_id: str, raw_request: Request):
    handler, registry = get_registry(raw_request)
    async with handler._fixed_prefix_lock:
        entry = registry.get(prefix_id)
        if entry is None:
            raise HTTPException(404, "Unknown prefix_id")
        if entry["active"]:
            raise HTTPException(409, "Prefix has active requests; retry later")
        result = await utility(handler, "unregister", prefix_id)
        del registry[prefix_id]
        return result


async def prefix_completion(raw_request: Request):
    # Read the body before the disconnect listener starts consuming ASGI events.
    payload = await raw_request.json()
    return await _prefix_completion(payload, raw_request=raw_request)


@with_cancellation
async def _prefix_completion(payload: dict, raw_request: Request):
    handler, registry = get_registry(raw_request)
    prefix_id = payload.get("prefix_id")
    entry = registry.get(prefix_id)
    if entry is None:
        raise HTTPException(404, "Unknown prefix_id; register after restart")
    allowed = {"prefix_id", "image_url", "uuid", "max_tokens", "temperature", "seed"}
    if set(payload) - allowed or "image_url" not in payload:
        raise HTTPException(400, "Unsupported fields or missing image_url")
    body = copy.deepcopy(entry["body"])
    for message in body["messages"]:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"] = payload["image_url"]
                    part["uuid"] = payload.get("uuid") or uuid.uuid4().hex
    body.update(
        {k: payload[k] for k in ("max_tokens", "temperature", "seed") if k in payload}
    )
    body["cache_salt"] = entry["salt"]
    body["vllm_xargs"] = {"fixed_prefix_id": prefix_id}
    body["stream"] = False
    request = ChatCompletionRequest.model_validate(body)
    entry["active"] += 1
    try:
        result = await handler.create_chat_completion(request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(result.model_dump(), status_code=result.error.code)
        if not isinstance(result, ChatCompletionResponse):
            raise RuntimeError("Unexpected streaming response")
        return JSONResponse(result.model_dump())
    finally:
        entry["active"] -= 1
