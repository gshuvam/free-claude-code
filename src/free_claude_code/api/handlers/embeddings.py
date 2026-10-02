from typing import Any

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.ports import ProviderResolver
from free_claude_code.config.settings import Settings
from free_claude_code.core.embeddings import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    generate_mock_embedding,
)


def _get_encoder() -> Any:
    from free_claude_code.core.token_estimation import initialize_token_estimation

    encoder = initialize_token_estimation()
    if encoder is not None:
        return encoder
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


class EmbeddingsHandler:
    """Handle OpenAI-compatible embedding requests."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
    ) -> None:
        self._settings = settings
        self._provider_resolver = provider_resolver

    async def create(self, request_data: EmbeddingRequest) -> EmbeddingResponse:
        encoder = _get_encoder()
        raw_input = request_data.input
        texts: list[str] = []
        if isinstance(raw_input, str):
            texts = [raw_input]
        elif isinstance(raw_input, list):
            if not raw_input:
                raise InvalidRequestError("input cannot be empty")
            first = raw_input[0]
            if isinstance(first, str):
                texts = [str(x) for x in raw_input]
            elif isinstance(first, int):
                tokens = [int(x) for x in raw_input if isinstance(x, int)]
                texts = [encoder.decode(tokens)]
            elif isinstance(first, list):
                texts = []
                for item in raw_input:
                    if isinstance(item, list):
                        tokens = [int(x) for x in item if isinstance(x, int)]
                        texts.append(encoder.decode(tokens))
                    else:
                        raise InvalidRequestError("input format is not supported")
            else:
                raise InvalidRequestError("input format is not supported")
        else:
            raise InvalidRequestError("input format is not supported")

        prompt_tokens = sum(
            len(encoder.encode(t, disallowed_special=())) for t in texts
        )

        requested_model = request_data.model
        if "/" in requested_model:
            provider_id, _, model_id = requested_model.partition("/")
            if provider_id == "nvidia":
                provider_id = "nvidia_nim"
                model_id = requested_model
        else:
            configured_model = self._settings.model_embedding
            provider_id, _, model_id = configured_model.partition("/")
            requested_model = configured_model

        target_dimensions = request_data.dimensions or 1536

        embeddings: list[list[float]] = []
        if provider_id == "mock":
            embeddings = [generate_mock_embedding(t, target_dimensions) for t in texts]
        else:
            provider = await self._provider_resolver(provider_id)
            get_embedding = getattr(provider, "get_embedding", None)
            if not callable(get_embedding):
                logger.warning(
                    "Provider '{}' does not support embeddings. Falling back to mock embeddings.",
                    provider_id,
                )
                embeddings = [
                    generate_mock_embedding(t, target_dimensions) for t in texts
                ]
            else:
                try:
                    extra_params = request_data.model_extra or {}
                    raw_embeddings = await get_embedding(
                        texts=texts,
                        model=model_id,
                        dimensions=request_data.dimensions,
                        **extra_params,
                    )
                    embeddings = []
                    for vec in raw_embeddings:
                        if len(vec) == target_dimensions:
                            embeddings.append(vec)
                        elif len(vec) > target_dimensions:
                            embeddings.append(vec[:target_dimensions])
                        else:
                            embeddings.append(
                                vec + [0.0] * (target_dimensions - len(vec))
                            )
                except NotImplementedError:
                    logger.warning(
                        "Provider '{}' does not support embeddings. Falling back to mock embeddings.",
                        provider_id,
                    )
                    embeddings = [
                        generate_mock_embedding(t, target_dimensions) for t in texts
                    ]

        data_items = [
            EmbeddingData(index=idx, embedding=emb)
            for idx, emb in enumerate(embeddings)
        ]
        usage = EmbeddingUsage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        )
        return EmbeddingResponse(
            data=data_items,
            model=requested_model,
            usage=usage,
        )
