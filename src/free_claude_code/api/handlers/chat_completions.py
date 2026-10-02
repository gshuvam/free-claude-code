"""OpenAI Chat Completions API adapter and stream converter."""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import JSONResponse, Response

from free_claude_code.api.models.chat_completions import (
    ChatCompletionRequest,
)
from free_claude_code.api.request_errors import (
    ordinary_application_error_response,
)
from free_claude_code.api.request_ids import new_request_id
from free_claude_code.api.response_streams import (
    OPENAI_CHAT_SSE_HEADERS,
    openai_chat_sse_streaming_response,
)
from free_claude_code.application.errors import ApplicationError, InvalidRequestError
from free_claude_code.core.anthropic import (
    ContentBlockImage,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
    MessagesRequest,
    Tool,
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from free_claude_code.core.diagnostics import safe_exception_message
from free_claude_code.core.failures import find_execution_failure
from free_claude_code.core.openai_responses import (
    openai_error_payload,
    openai_error_type_for_failure,
)

from .messages import MessagesHandler


def chat_completion_to_messages_request(
    request_data: ChatCompletionRequest,
) -> MessagesRequest:
    """Convert an incoming OpenAI ChatCompletionRequest into an Anthropic MessagesRequest."""
    system_parts: list[str] = []
    converted_messages: list[Message] = []

    for msg in request_data.messages:
        role = msg.role.lower()
        if role in ("system", "developer"):
            if isinstance(msg.content, str) and msg.content.strip():
                system_parts.append(msg.content.strip())
            elif isinstance(msg.content, list):
                system_parts.extend(
                    str(part.get("text", "")).strip()
                    for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            continue

        if role in ("tool", "function"):
            tool_call_id = msg.tool_call_id or ""
            content_str = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content or "", ensure_ascii=False)
            )
            block = ContentBlockToolResult(
                type="tool_result",
                tool_use_id=tool_call_id,
                content=content_str,
            )
            if converted_messages and converted_messages[-1].role == "user":
                last_msg = converted_messages[-1]
                if isinstance(last_msg.content, list):
                    last_msg.content.append(block)
                else:
                    last_msg.content = [
                        ContentBlockText(type="text", text=last_msg.content),
                        block,
                    ]
            else:
                converted_messages.append(Message(role="user", content=[block]))
            continue

        if role == "assistant":
            blocks: list[Any] = []
            if isinstance(msg.content, str) and msg.content:
                blocks.append(ContentBlockText(type="text", text=msg.content))
            elif isinstance(msg.content, list):
                blocks.extend(
                    ContentBlockText(type="text", text=str(part.get("text", "")))
                    for part in msg.content
                    if isinstance(part, dict) and part.get("type") == "text"
                )

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", "{}")
                    parsed_args: dict[str, Any]
                    if isinstance(fn_args, str):
                        try:
                            val = json.loads(fn_args)
                            parsed_args = (
                                val if isinstance(val, dict) else {"raw": fn_args}
                            )
                        except Exception:
                            parsed_args = {"raw": fn_args}
                    elif isinstance(fn_args, dict):
                        parsed_args = fn_args
                    else:
                        parsed_args = {}
                    blocks.append(
                        ContentBlockToolUse(
                            type="tool_use",
                            id=tc_id,
                            name=fn_name,
                            input=parsed_args,
                        )
                    )
            if not blocks:
                blocks.append(ContentBlockText(type="text", text=""))

            if converted_messages and converted_messages[-1].role == "assistant":
                last_msg = converted_messages[-1]
                if isinstance(last_msg.content, list):
                    last_msg.content.extend(blocks)
                else:
                    last_msg.content = [
                        ContentBlockText(type="text", text=last_msg.content),
                        *blocks,
                    ]
            else:
                converted_messages.append(Message(role="assistant", content=blocks))
            continue

        if role == "user":
            user_blocks: list[Any] = []
            if isinstance(msg.content, str):
                user_blocks.append(ContentBlockText(type="text", text=msg.content))
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        user_blocks.append(
                            ContentBlockText(
                                type="text", text=str(part.get("text", ""))
                            )
                        )
                    elif ptype == "image_url":
                        url_obj = part.get("image_url", {})
                        url = (
                            url_obj.get("url", "")
                            if isinstance(url_obj, dict)
                            else str(url_obj)
                        )
                        if url.startswith("data:"):
                            header, _, data = url.partition(",")
                            media_type = (
                                header.replace("data:", "")
                                .replace(";base64", "")
                                .strip()
                            )
                            user_blocks.append(
                                ContentBlockImage(
                                    type="image",
                                    source={
                                        "type": "base64",
                                        "media_type": media_type or "image/jpeg",
                                        "data": data,
                                    },
                                )
                            )
                        else:
                            user_blocks.append(
                                ContentBlockText(type="text", text=f"[Image: {url}]")
                            )
            if not user_blocks:
                user_blocks.append(ContentBlockText(type="text", text=""))

            if converted_messages and converted_messages[-1].role == "user":
                last_msg = converted_messages[-1]
                if isinstance(last_msg.content, list):
                    last_msg.content.extend(user_blocks)
                else:
                    last_msg.content = [
                        ContentBlockText(type="text", text=last_msg.content),
                        *user_blocks,
                    ]
            else:
                converted_messages.append(Message(role="user", content=user_blocks))

    if not converted_messages:
        converted_messages.append(Message(role="user", content="Hello"))
    elif converted_messages[0].role != "user":
        converted_messages.insert(0, Message(role="user", content=" "))

    converted_tools: list[Tool] | None = None
    if request_data.tools:
        converted_tools = []
        for tool_item in request_data.tools:
            if not isinstance(tool_item, dict):
                continue
            fn = tool_item.get("function")
            if isinstance(fn, dict):
                converted_tools.append(
                    Tool(
                        name=fn.get("name", ""),
                        description=fn.get("description"),
                        input_schema=fn.get("parameters") or {"type": "object"},
                    )
                )
            elif "name" in tool_item:
                converted_tools.append(
                    Tool(
                        name=tool_item.get("name", ""),
                        description=tool_item.get("description"),
                        input_schema=tool_item.get("parameters")
                        or tool_item.get("input_schema")
                        or {"type": "object"},
                    )
                )

    converted_tool_choice: dict[str, Any] | None = None
    if request_data.tool_choice:
        if isinstance(request_data.tool_choice, str):
            if request_data.tool_choice == "auto":
                converted_tool_choice = {"type": "auto"}
            elif request_data.tool_choice in ("required", "any"):
                converted_tool_choice = {"type": "any"}
            elif request_data.tool_choice == "none":
                converted_tool_choice = None
                converted_tools = None
        elif isinstance(request_data.tool_choice, dict):
            fn = request_data.tool_choice.get("function")
            if isinstance(fn, dict) and "name" in fn:
                converted_tool_choice = {"type": "tool", "name": fn["name"]}
            elif "name" in request_data.tool_choice:
                converted_tool_choice = {
                    "type": "tool",
                    "name": request_data.tool_choice["name"],
                }

    max_tokens = (
        request_data.max_completion_tokens
        if request_data.max_completion_tokens is not None
        else (request_data.max_tokens if request_data.max_tokens is not None else 4096)
    )

    stop_sequences: list[str] | None = None
    if request_data.stop:
        if isinstance(request_data.stop, str):
            stop_sequences = [request_data.stop]
        elif isinstance(request_data.stop, list):
            stop_sequences = [str(s) for s in request_data.stop]

    return MessagesRequest(
        model=request_data.model,
        messages=converted_messages,
        system="\n\n".join(system_parts) if system_parts else None,
        max_tokens=max_tokens,
        stream=bool(request_data.stream),
        temperature=request_data.temperature,
        top_p=request_data.top_p,
        tools=converted_tools,
        tool_choice=converted_tool_choice,
        stop_sequences=stop_sequences,
    )


def anthropic_message_to_chat_completion(
    message: dict[str, Any],
    *,
    model: str,
    request_id: str,
) -> dict[str, Any]:
    """Convert an aggregated Anthropic message dictionary to an OpenAI chat completion JSON response."""
    chat_id = f"chatcmpl-{request_id.replace('req_', '')}"
    created = int(time.time())

    content_text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in message.get("content", []):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            content_text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": (
                            json.dumps(block.get("input", {}), ensure_ascii=False)
                            if isinstance(block.get("input"), dict)
                            else str(block.get("input") or "{}")
                        ),
                    },
                }
            )

    content_str: str | None = (
        "".join(content_text_parts)
        if content_text_parts
        else (None if tool_calls else "")
    )

    msg_obj: dict[str, Any] = {
        "role": "assistant",
        "content": content_str,
    }
    if tool_calls:
        msg_obj["tool_calls"] = tool_calls
    if reasoning_parts:
        msg_obj["reasoning_content"] = "".join(reasoning_parts)

    stop_reason = message.get("stop_reason")
    finish_reason = "stop"
    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason in ("end_turn", "stop_sequence"):
        finish_reason = "stop"

    usage = message.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    completion_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": msg_obj,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def anthropic_to_openai_chat_stream(
    stream: AsyncIterator[str],
    *,
    model: str,
    request_id: str,
) -> AsyncIterator[str]:
    """Convert an Anthropic SSE stream into OpenAI Chat Completions chunk SSE events."""
    decoder = AnthropicSSEDecoder()
    chat_id = f"chatcmpl-{request_id.replace('req_', '')}"
    created = int(time.time())
    started = False
    active_tools: dict[int, dict[str, Any]] = {}
    tool_counter = 0

    try:
        async for chunk in stream:
            events = decoder.feed(chunk)
            for event in events:
                ptype = event.event
                if ptype == "ping":
                    continue
                payload = event.data
                if not isinstance(payload, dict):
                    continue

                if ptype == "message_start":
                    if not started:
                        started = True
                        chunk_obj = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_obj)}\n\n"

                elif ptype == "content_block_start":
                    idx = payload.get("index", 0)
                    block = payload.get("content_block", {})
                    if block.get("type") == "tool_use":
                        t_idx = tool_counter
                        tool_counter += 1
                        active_tools[idx] = {
                            "tool_index": t_idx,
                            "id": block.get("id", ""),
                        }
                        chunk_obj = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": t_idx,
                                                "id": block.get("id", ""),
                                                "type": "function",
                                                "function": {
                                                    "name": block.get("name", ""),
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_obj)}\n\n"

                elif ptype == "content_block_delta":
                    idx = payload.get("index", 0)
                    delta = payload.get("delta", {})
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            chunk_obj = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_obj)}\n\n"
                    elif dtype == "thinking_delta":
                        thinking = delta.get("thinking", "")
                        if thinking:
                            chunk_obj = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"reasoning_content": thinking},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_obj)}\n\n"
                    elif dtype == "input_json_delta":
                        partial_json = delta.get("partial_json", "")
                        tool_meta = active_tools.get(idx)
                        t_idx = tool_meta["tool_index"] if tool_meta else 0
                        chunk_obj = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": t_idx,
                                                "function": {"arguments": partial_json},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_obj)}\n\n"

                elif ptype == "message_delta":
                    delta = payload.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    finish_reason = "stop"
                    if stop_reason == "tool_use":
                        finish_reason = "tool_calls"
                    elif stop_reason == "max_tokens":
                        finish_reason = "length"
                    elif stop_reason in ("end_turn", "stop_sequence"):
                        finish_reason = "stop"

                    chunk_obj = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    usage = payload.get("usage")
                    if usage and isinstance(usage, dict):
                        in_tok = usage.get("input_tokens", 0)
                        out_tok = usage.get("output_tokens", 0)
                        chunk_obj["usage"] = {
                            "prompt_tokens": in_tok,
                            "completion_tokens": out_tok,
                            "total_tokens": in_tok + out_tok,
                        }
                    yield f"data: {json.dumps(chunk_obj)}\n\n"

                elif ptype == "message_stop":
                    yield "data: [DONE]\n\n"
                    return

                elif ptype == "error":
                    err_obj = payload.get("error", {})
                    err_chunk = {
                        "error": {
                            "message": err_obj.get("message", "Stream error"),
                            "type": err_obj.get("type", "api_error"),
                        }
                    }
                    yield f"data: {json.dumps(err_chunk)}\n\ndata: [DONE]\n\n"
                    return

        yield "data: [DONE]\n\n"
    finally:
        pass


class ChatCompletionsHandler:
    """Handle OpenAI-compatible Chat Completions requests."""

    def __init__(self, messages_handler: MessagesHandler) -> None:
        self._messages_handler = messages_handler

    async def create(
        self, request_data: ChatCompletionRequest, *, request_id: str | None = None
    ) -> Response:
        request_id = request_id or new_request_id()
        if not request_data.model.strip():
            raise InvalidRequestError("Chat completion model must not be empty.")
        if not request_data.messages:
            raise InvalidRequestError("Chat completion messages must not be empty.")

        messages_request = chat_completion_to_messages_request(request_data)
        stream, resolved_model = await self._messages_handler.execute_messages_stream(
            messages_request, request_id=request_id
        )

        if not request_data.stream:
            message, error, _complete = await aggregate_anthropic_sse_to_message(stream)
            if error is not None:
                err_msg = error.get("message", "Upstream provider error")
                err_type = error.get("type", "api_error")
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": err_msg, "type": err_type}},
                )
            return JSONResponse(
                content=anthropic_message_to_chat_completion(
                    message,
                    model=resolved_model,
                    request_id=request_id,
                )
            )

        chat_stream = anthropic_to_openai_chat_stream(
            stream,
            model=resolved_model,
            request_id=request_id,
        )
        return await openai_chat_sse_streaming_response(
            chat_stream,
            headers=OPENAI_CHAT_SSE_HEADERS,
            pre_start_error_response=lambda exc: self._pre_start_error_response(
                exc, request_id=request_id
            ),
            request_id=request_id,
        )

    def _pre_start_error_response(
        self, exc: BaseException, *, request_id: str
    ) -> Response:
        if isinstance(exc, ApplicationError):
            return ordinary_application_error_response(
                exc, wire_api="chat_completions", request_id=request_id
            )
        failure = find_execution_failure(exc)
        if failure is not None:
            return JSONResponse(
                status_code=failure.status_code,
                content=openai_error_payload(
                    message=failure.message,
                    error_type=openai_error_type_for_failure(failure.kind),
                ),
            )
        return JSONResponse(
            status_code=500,
            content=openai_error_payload(
                message=safe_exception_message(exc),
                error_type="api_error",
            ),
        )
