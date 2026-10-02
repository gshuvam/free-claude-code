"""Tests for OpenAI Chat Completions API adapter and single model endpoint."""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderModelInfo
from tests.api.support import create_test_app


class MockStreamProvider(BaseProvider):
    """A mock provider that yields predefined Anthropic SSE events."""

    def __init__(self, events: list[tuple[str, dict]] | None = None) -> None:
        self.events = events or [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test_123",
                        "type": "message",
                        "role": "assistant",
                        "model": "test-provider/test-model",
                        "content": [],
                        "usage": {"input_tokens": 10, "output_tokens": 1},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello from mock!"},
                },
            ),
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
            ),
            (
                "message_stop",
                {"type": "message_stop"},
            ),
        ]
        self.received_messages_requests: list[MessagesRequest] = []

    def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        self.received_messages_requests.append(request)

        async def _generator():
            for event_type, payload in self.events:
                yield format_sse_event(event_type, payload)

        return _generator()

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        async def _empty():
            if False:
                yield ""

        return _empty()

    async def cleanup(self) -> None:
        pass

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset()


def test_chat_completions_non_streaming():
    provider = MockStreamProvider()
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello!"},
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello from mock!"
        assert choice["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5
        assert data["usage"]["total_tokens"] == 15

        # Check conversion to MessagesRequest
        assert len(provider.received_messages_requests) == 1
        msg_req = provider.received_messages_requests[0]
        assert msg_req.system == "You are helpful"
        assert len(msg_req.messages) == 1
        assert msg_req.messages[0].role == "user"


def test_chat_completions_streaming():
    provider = MockStreamProvider()
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "Stream to me"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        parsed_chunks: list[dict[str, Any]] = []
        done_seen = False
        for line in lines:
            if line.startswith("data: "):
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    done_seen = True
                else:
                    parsed_chunks.append(json.loads(data_str))

        assert len(parsed_chunks) >= 3
        # First chunk is role chunk
        assert parsed_chunks[0]["object"] == "chat.completion.chunk"
        assert parsed_chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Content chunk
        assert any(
            c["choices"][0]["delta"].get("content") == "Hello from mock!"
            for c in parsed_chunks
        )

        # Finish chunk
        assert any(c["choices"][0]["finish_reason"] == "stop" for c in parsed_chunks)

        # [DONE] was seen
        assert done_seen


def test_chat_completions_tool_calling_non_streaming():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_tool_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "openai/test-model",
                    "content": [],
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_weather_1",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"},
                },
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    provider = MockStreamProvider(events=events)
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "What is the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                            },
                        },
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert "tool_calls" in choice["message"]
        tool_call = choice["message"]["tool_calls"][0]
        assert tool_call["id"] == "call_weather_1"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {
            "location": "San Francisco"
        }


def test_chat_completions_tool_calling_streaming():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_tool_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "openai/test-model",
                    "content": [],
                    "usage": {"input_tokens": 15, "output_tokens": 2},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_calc_1",
                    "name": "calculator",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"expr": "2+2"}',
                },
            },
        ),
        (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 12},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    provider = MockStreamProvider(events=events)
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "Calculate 2+2"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        chunks = [
            json.loads(line[len("data: ") :])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

        # Look for tool call chunks
        tc_chunks = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
        assert len(tc_chunks) >= 2
        assert (
            tc_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "calculator"
        )
        assert (
            tc_chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"][
                "arguments"
            ]
            == '{"expr": "2+2"}'
        )


def test_chat_completions_reasoning_streaming():
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_think",
                    "type": "message",
                    "role": "assistant",
                    "model": "openai/test-model",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think deeply"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Answer"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    provider = MockStreamProvider(events=events)
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "Think"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        chunks = [
            json.loads(line[len("data: ") :])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        reasoning_chunks = [
            c for c in chunks if "reasoning_content" in c["choices"][0]["delta"]
        ]
        assert len(reasoning_chunks) >= 1
        assert (
            reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"]
            == "Let me think deeply"
        )


def test_chat_completions_invalid_requests():
    provider = MockStreamProvider()
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        # Empty model
        res = client.post(
            "/v1/chat/completions",
            json={"model": "", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code == 400
        assert res.json()["error"]["type"] == "invalid_request_error"

        # Empty messages
        res2 = client.post(
            "/v1/chat/completions",
            json={"model": "openai/test-model", "messages": []},
        )
        assert res2.status_code == 400
        assert res2.json()["error"]["type"] == "invalid_request_error"


def test_chat_completions_probe():
    provider = MockStreamProvider()
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app) as client:
        res = client.options("/v1/chat/completions")
        assert res.status_code == 204
        assert "POST" in res.headers["allow"]

        res_head = client.head("/v1/chat/completions")
        assert res_head.status_code == 204


def test_get_model_endpoint():
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings)

    with TestClient(app) as client:
        # Known claude model from default catalog
        res = client.get("/v1/models/claude-sonnet-4-20250514")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == "claude-sonnet-4-20250514"
        assert data["object"] == "model"

        # Configured model
        res_cfg = client.get("/v1/models/test-model")
        assert res_cfg.status_code == 200
        assert res_cfg.json()["id"] == "test-model"

        # Non-existent model -> 404 with OpenAI error payload
        res_404 = client.get("/v1/models/non-existent-fake-model")
        assert res_404.status_code == 404
        err_data = res_404.json()
        assert "error" in err_data
        assert err_data["error"]["code"] == "model_not_found"
        assert err_data["error"]["type"] == "invalid_request_error"

        # Probe on /v1/models/{model_id}
        res_opts = client.options("/v1/models/test-model")
        assert res_opts.status_code == 204
        assert "GET" in res_opts.headers["allow"]


def test_chat_completions_general_error_openai_formatting():
    class BrokenProvider(MockStreamProvider):
        def stream_messages(self, *args, **kwargs):
            raise RuntimeError("Unexpected boom in provider")

    provider = BrokenProvider()
    settings = Settings(model="openai/test-model")
    app = create_test_app(settings, providers={"openai": provider})

    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert res.status_code == 500
        data = res.json()
        assert "error" in data
        assert data["error"]["type"] == "api_error"
        assert "boom" in data["error"]["message"]
