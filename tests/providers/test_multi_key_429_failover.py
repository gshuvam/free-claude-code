"""Tests for comma-separated API key round-robin failover on 429 errors."""

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.api_key_pool import ApiKeyPool
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission, make_provider_config

pytestmark = pytest.mark.asyncio


class Wire(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes | Exception],
        *,
        closed: Callable[[], None] | None = None,
    ) -> None:
        self.chunks = chunks
        self.closed = False
        self._on_close = closed

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self) -> None:
        self.closed = True
        if self._on_close is not None:
            self._on_close()


class NativeEndpoint:
    def __init__(self) -> None:
        self.base_url = "https://native.invalid/v1/"

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        return HttpEndpoint(
            self.base_url,
            {
                "x-api-key": "initial",
                "anthropic-version": "2023-06-01",
            },
        )


def _anthropic_sse(*events: JsonObject) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode()


def _anthropic_events(text: str = "hello") -> list[JsonObject]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "upstream-id",
                "type": "message",
                "model": "claude-3-opus",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


def _openai_chat_chunk(text: str = "hello") -> httpx2.Response:
    chunk = {
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}],
    }
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
    )


async def test_openai_multi_key_failover() -> None:
    """Test OpenAI-compatible stream: first key gets 429, rotates, second key succeeds."""
    pool = ApiKeyPool("key1,key2")
    config = replace(
        make_provider_config(api_key="key1", base_url="https://provider.invalid/v1"),
        api_key_pool=pool,
    )
    auth_headers: list[str] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers.get("authorization", "")
        auth_headers.append(auth)
        if auth == "Bearer key1":
            return httpx2.Response(429, json={"error": {"message": "Rate limited"}})
        if auth == "Bearer key2":
            return _openai_chat_chunk("hello from key2")
        return httpx2.Response(400, json={"error": {"message": "Unknown key"}})

    client = AsyncOpenAI(
        api_key="key1",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    provider = OpenAIChatProvider(
        config,
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.DISABLED),
            NO_REASONING,
        ),
        admission=immediate_admission(max_attempts=3),
        client=client,
    )
    try:
        events = [
            event
            async for event in provider.stream_messages(
                make_messages_request("test-model")
            )
        ]
        assert "hello from key2" in "".join(events)
        assert auth_headers == ["Bearer key1", "Bearer key2"]
        assert provider._api_key == "key2"
    finally:
        await provider.cleanup()


async def test_openai_multi_key_all_fail() -> None:
    """Test OpenAI-compatible stream: all keys get 429, raises ExecutionFailure."""
    pool = ApiKeyPool("key1,key2")
    config = replace(
        make_provider_config(api_key="key1", base_url="https://provider.invalid/v1"),
        api_key_pool=pool,
    )
    auth_headers: list[str] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers.get("authorization", "")
        auth_headers.append(auth)
        return httpx2.Response(
            429,
            json={
                "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
            },
        )

    client = AsyncOpenAI(
        api_key="key1",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    provider = OpenAIChatProvider(
        config,
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.DISABLED),
            NO_REASONING,
        ),
        admission=immediate_admission(max_attempts=1),
        client=client,
    )
    try:
        with pytest.raises(ExecutionFailure):
            _ = [
                event
                async for event in provider.stream_messages(
                    make_messages_request("test-model")
                )
            ]
        assert auth_headers == ["Bearer key1", "Bearer key2"]
    finally:
        await provider.cleanup()


async def test_openai_multi_key_no_rotation_on_400() -> None:
    """Test OpenAI-compatible stream: 400 Bad Request error does not trigger rotation."""
    pool = ApiKeyPool("key1,key2")
    config = replace(
        make_provider_config(api_key="key1", base_url="https://provider.invalid/v1"),
        api_key_pool=pool,
    )
    auth_headers: list[str] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers.get("authorization", "")
        auth_headers.append(auth)
        return httpx2.Response(
            400,
            json={"error": {"message": "Bad request", "type": "invalid_request_error"}},
        )

    client = AsyncOpenAI(
        api_key="key1",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    provider = OpenAIChatProvider(
        config,
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.DISABLED),
            NO_REASONING,
        ),
        admission=immediate_admission(max_attempts=3),
        client=client,
    )
    try:
        with pytest.raises(ExecutionFailure):
            _ = [
                event
                async for event in provider.stream_messages(
                    make_messages_request("test-model")
                )
            ]
        assert auth_headers == ["Bearer key1"]
        assert provider._api_key == "key1"
    finally:
        await provider.cleanup()


async def test_openai_embeddings_multi_key_failover() -> None:
    """Test get_embedding rotates API key on 429 error and succeeds on subsequent key."""
    pool = ApiKeyPool("key1,key2")
    config = replace(
        make_provider_config(api_key="key1", base_url="https://provider.invalid/v1"),
        api_key_pool=pool,
    )
    auth_headers: list[str] = []

    def reply(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers.get("authorization", "")
        auth_headers.append(auth)
        if auth == "Bearer key1":
            return httpx2.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limit reached",
                        "type": "rate_limit_error",
                    }
                },
            )
        if auth == "Bearer key2":
            return httpx2.Response(
                200,
                json={
                    "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                    "model": "text-embedding-test",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        return httpx2.Response(400, json={"error": {"message": "Unknown"}})

    client = AsyncOpenAI(
        api_key="key1",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
    )
    provider = OpenAIChatProvider(
        config,
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.DISABLED),
            NO_REASONING,
        ),
        admission=immediate_admission(max_attempts=3),
        client=client,
    )
    try:
        embeddings = await provider.get_embedding(
            ["sample text"], "text-embedding-test"
        )
        assert embeddings == [[0.1, 0.2, 0.3]]
        assert auth_headers == ["Bearer key1", "Bearer key2"]
        assert provider._api_key == "key2"
    finally:
        await provider.cleanup()


async def test_anthropic_multi_key_failover() -> None:
    """Test AnthropicMessagesTransport: first key gets 429, rotates, second key succeeds."""
    pool = ApiKeyPool("key1,key2")
    received_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.headers.get("x-api-key", "")
        received_keys.append(api_key)
        if api_key == "key1":
            return httpx.Response(
                429,
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "limit"},
                },
            )
        if api_key == "key2":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Wire([_anthropic_sse(*_anthropic_events("hello native key2"))]),
            )
        return httpx.Response(400)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = AnthropicMessagesTransport(
            client=client,
            admission=immediate_admission(max_attempts=3),
            provider_name="TEST_ANTHROPIC",
            replay_scope="test/messages",
            read_timeout_s=3,
            api_key="key1",
            api_key_pool=pool,
        )
        events = [
            event
            async for event in transport.stream_messages(
                make_messages_request("claude-3-opus"),
                endpoint_context=NativeEndpoint(),
            )
        ]
        assert "hello native key2" in "".join(events)
        assert received_keys == ["key1", "key2"]
        assert transport._api_key == "key2"


async def test_anthropic_multi_key_all_fail() -> None:
    """Test AnthropicMessagesTransport: all keys get 429, raises ExecutionFailure."""
    pool = ApiKeyPool("key1,key2")
    received_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.headers.get("x-api-key", "")
        received_keys.append(api_key)
        return httpx.Response(
            429,
            json={
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "limit"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = AnthropicMessagesTransport(
            client=client,
            admission=immediate_admission(max_attempts=1),
            provider_name="TEST_ANTHROPIC",
            replay_scope="test/messages",
            read_timeout_s=3,
            api_key="key1",
            api_key_pool=pool,
        )
        with pytest.raises(ExecutionFailure):
            _ = [
                event
                async for event in transport.stream_messages(
                    make_messages_request("claude-3-opus"),
                    endpoint_context=NativeEndpoint(),
                )
            ]
        assert received_keys == ["key1", "key2"]


async def test_anthropic_multi_key_no_rotation_on_400() -> None:
    """Test AnthropicMessagesTransport: 400 error does not rotate key."""
    pool = ApiKeyPool("key1,key2")
    received_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.headers.get("x-api-key", "")
        received_keys.append(api_key)
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = AnthropicMessagesTransport(
            client=client,
            admission=immediate_admission(max_attempts=3),
            provider_name="TEST_ANTHROPIC",
            replay_scope="test/messages",
            read_timeout_s=3,
            api_key="key1",
            api_key_pool=pool,
        )
        with pytest.raises(ExecutionFailure):
            _ = [
                event
                async for event in transport.stream_messages(
                    make_messages_request("claude-3-opus"),
                    endpoint_context=NativeEndpoint(),
                )
            ]
        assert received_keys == ["key1"]
        assert transport._api_key == "key1"
