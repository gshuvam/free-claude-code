"""OpenAI-compatible embedding models and deterministic mock generator."""

import hashlib
import math
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def generate_mock_embedding(text: str, dimensions: int) -> list[float]:
    """Generate a deterministic, unit-length normalized mock embedding vector."""
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    vector = [rng.gauss(0, 1) for _ in range(dimensions)]
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector


_generate_mock_embedding = generate_mock_embedding


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str | list[str] | list[int] | list[list[int]]
    model: str
    dimensions: int | None = Field(default=None, ge=1)
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
