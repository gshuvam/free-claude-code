import math
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.embeddings import _generate_mock_embedding
from free_claude_code.providers.base import BaseProvider
from tests.api.support import create_test_app


@pytest.fixture
def client():
    """HTTP client with test app dependencies."""
    app = create_test_app(Settings())
    with TestClient(app) as test_client:
        yield test_client


def test_mock_embedding_generation():
    """Verify deterministic mock embedding generation and normalization."""
    text = "hello world"
    vec1 = _generate_mock_embedding(text, 1536)
    vec2 = _generate_mock_embedding(text, 1536)
    assert vec1 == vec2
    assert len(vec1) == 1536

    # Verify unit-length normalization
    norm = math.sqrt(sum(x * x for x in vec1))
    assert pytest.approx(norm) == 1.0


def test_embedding_endpoint_mock_single(client: TestClient):
    """Test single string input on the embedding endpoint."""
    payload = {
        "model": "mock/embedding-mock",
        "input": "test input",
        "dimensions": 10,
    }
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["index"] == 0
    assert len(data["data"][0]["embedding"]) == 10
    assert data["model"] == "mock/embedding-mock"
    assert data["usage"]["prompt_tokens"] > 0


def test_embedding_endpoint_mock_batch(client: TestClient):
    """Test batch string inputs on the embeddings endpoint."""
    payload = {
        "model": "mock/embedding-mock",
        "input": ["test input 1", "test input 2"],
        "dimensions": 20,
    }
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0
    assert data["data"][1]["index"] == 1
    assert len(data["data"][0]["embedding"]) == 20
    assert len(data["data"][1]["embedding"]) == 20


def test_embedding_endpoint_mock_tokens_input(client: TestClient):
    """Test token IDs (list of int) as input."""
    payload = {
        "model": "mock/embedding-mock",
        "input": [1000, 2000, 3000],
        "dimensions": 5,
    }
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 5


def test_embedding_endpoint_mock_batched_tokens_input(client: TestClient):
    """Test token IDs batch (list of list of int) as input."""
    payload = {
        "model": "mock/embedding-mock",
        "input": [[1000, 2000], [3000, 4000, 5000]],
        "dimensions": 8,
    }
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == 8
    assert len(data["data"][1]["embedding"]) == 8


def test_provider_embedding():
    """Test routing to an active provider and proxy dimension processing (truncation/padding)."""
    mock_prov = MagicMock(spec=BaseProvider)

    async def _mock_get_embedding(texts, model, dimensions=None, **kwargs):
        return [[0.1, 0.2, 0.3]]

    mock_prov.get_embedding = _mock_get_embedding

    app = create_test_app(Settings(), providers={"nvidia_nim": mock_prov})
    with TestClient(app) as client:
        payload = {
            "model": "nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input": "test input",
            "dimensions": 3,
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

        # 2. Test proxy dimension truncation
        payload_truncate = {
            "model": "nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input": "test input",
            "dimensions": 2,
        }
        response_truncate = client.post("/v1/embeddings", json=payload_truncate)
        assert response_truncate.status_code == 200
        data_truncate = response_truncate.json()
        assert data_truncate["data"][0]["embedding"] == [0.1, 0.2]

        # 3. Test proxy dimension padding
        payload_pad = {
            "model": "nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input": "test input",
            "dimensions": 5,
        }
        response_pad = client.post("/v1/embeddings", json=payload_pad)
        assert response_pad.status_code == 200
        data_pad = response_pad.json()
        assert data_pad["data"][0]["embedding"] == [0.1, 0.2, 0.3, 0.0, 0.0]


def test_provider_embedding_extra_kwargs():
    """Test routing to active provider and forwarding extra parameters."""
    mock_prov = MagicMock(spec=BaseProvider)

    async def _mock_get_embedding(texts, model, dimensions=None, **kwargs):
        assert kwargs.get("input_type") == "query"
        assert kwargs.get("another_custom_param") == "value"
        return [[0.1, 0.2]]

    mock_prov.get_embedding = _mock_get_embedding

    app = create_test_app(Settings(), providers={"nvidia_nim": mock_prov})
    with TestClient(app) as client:
        payload = {
            "model": "nvidia_nim/nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input": "test input",
            "dimensions": 2,
            "input_type": "query",
            "another_custom_param": "value",
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["embedding"] == [0.1, 0.2]


def test_provider_embedding_nvidia_prefix_mapping():
    """Test that model prefixed with 'nvidia/' is routed to 'nvidia_nim' with original ID."""
    mock_prov = MagicMock(spec=BaseProvider)

    async def _mock_get_embedding(texts, model, dimensions=None, **kwargs):
        assert model == "nvidia/llama-nemotron-embed-vl-1b-v2"
        return [[0.5, 0.6]]

    mock_prov.get_embedding = _mock_get_embedding

    app = create_test_app(Settings(), providers={"nvidia_nim": mock_prov})
    with TestClient(app) as client:
        payload = {
            "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
            "input": "test input",
            "dimensions": 2,
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["data"][0]["embedding"] == [0.5, 0.6]
