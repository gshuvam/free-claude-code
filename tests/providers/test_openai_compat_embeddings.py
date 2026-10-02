from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.nvidia_nim.client import NvidiaNimProvider
from tests.providers.support import immediate_admission


@pytest.mark.asyncio
async def test_openai_compat_get_embedding_extra_body():
    """Verify that OpenAIChatTransport passes extra parameters in extra_body."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        http_read_timeout=60.0,
        http_write_timeout=60.0,
        http_connect_timeout=10.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    provider = NvidiaNimProvider(
        config,
        nim_settings=NimSettings(),
        admission=immediate_admission(provider_name="nvidia_nim"),
    )

    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_item]

    with patch.object(
        provider._client.embeddings,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_create:
        res = await provider.get_embedding(
            texts=["hello"],
            model="nvidia/llama-nemotron-embed-vl-1b-v2",
            dimensions=1024,
            input_type="query",
            truncate="NONE",
        )

        assert res == [[0.1, 0.2, 0.3]]
        mock_create.assert_called_once_with(
            input=["hello"],
            model="nvidia/llama-nemotron-embed-vl-1b-v2",
            dimensions=1024,
            extra_body={"input_type": "query", "truncate": "NONE"},
        )
