# Free Claude Code (FCC) API Reference

This document provides a complete guide to all HTTP endpoints exposed by the Free Claude Code proxy server. AI agents and client libraries can use this reference to discover available models, select the exact model identifier, and invoke chat, response, and embedding APIs.

---

## 1. Quick Start & Server Conventions

- **Default Base URL**: `http://localhost:8082`
- **OpenAI Compatible Base URL**: `http://localhost:8082/v1`
- **Anthropic Compatible Base URL**: `http://localhost:8082`
- **Authentication**:
  - If `FCC_PROXY_TOKEN` (or `PROXY_TOKEN`) is set in the environment:
    - OpenAI endpoints: `Authorization: Bearer <TOKEN>`
    - Anthropic endpoints: `x-api-key: <TOKEN>` or `Authorization: Bearer <TOKEN>`
  - If no proxy token is set, authentication headers are optional.

---

## 2. Model Discovery Endpoints

Use these endpoints to find exact model identifiers that should be passed in the `"model"` field of API requests.

### 2.1 List All Models (`GET /v1/models`)
Returns the model catalog in standard OpenAI list format.

- **URL**: `GET /v1/models`
- **Query Parameters**:
  - `view`: Filters the catalog projection.
    - `view=claude` *(default)*: Lists Claude model aliases (e.g. `claude-sonnet-4-20250514`, `claude-3-7-sonnet-20250219`).
    - `view=responses`: Lists models exposed for OpenAI Responses and direct coding clients.
    - `view=messages`: Lists models for native Anthropic Messages.
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/v1/models?view=responses
  ```
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "openai/gpt-4o",
        "object": "model",
        "display_name": "GPT-4o",
        "created_at": "1970-01-01T00:00:00Z",
        "contextWindow": 128000,
        "maxCompletionTokens": 16384,
        "supportsReasoning": false
      },
      {
        "id": "deepseek/deepseek-chat",
        "object": "model",
        "display_name": "DeepSeek Chat",
        "created_at": "1970-01-01T00:00:00Z"
      }
    ],
    "default_model_id": "openai/gpt-4o"
  }
  ```
> **Note for AI Agents**: Use the string from the `"id"` attribute directly in the `"model"` field of your completion payloads.

---

### 2.2 Retrieve a Single Model (`GET /v1/models/{model_id}`)
Checks if a specific model identifier exists and retrieves its capabilities and limits.

- **URL**: `GET /v1/models/{model_id}`
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/v1/models/deepseek/deepseek-chat
  ```
- **Responses**:
  - `200 OK`: Model details object.
  - `404 Not Found`: Returns standard OpenAI error payload:
    ```json
    {
      "error": {
        "message": "The model 'invalid-model' does not exist",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found"
      }
    }
    ```

---

### 2.3 Live Provider Model Discovery (`GET /admin/api/models`)
Retrieves all models actively discovered across configured providers and connected local/remote backends (Ollama, LM Studio, Groq, OpenRouter, etc.).

- **URL**: `GET /admin/api/models`
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/admin/api/models
  ```
- **Example Response**:
  ```json
  {
    "models": [
      "openai/gpt-4o",
      "deepseek/deepseek-chat",
      "groq/llama-3.3-70b-versatile"
    ],
    "model_labels": {
      "openai/gpt-4o": "GPT-4o",
      "deepseek/deepseek-chat": "DeepSeek Chat"
    },
    "failed_providers": []
  }
  ```

---

### 2.4 Muse Code Model Catalog (`GET /muse-code/models`)
Returns the model catalog formatted specifically for Muse Code with context and token limits embedded in the model metadata.

- **URL**: `GET /muse-code/models`
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/muse-code/models
  ```

---

### 2.5 Refresh Provider Model Catalog (`POST /admin/api/models/refresh`)
Triggers an upstream refresh from providers (such as scanning local Ollama or remote provider endpoints) and returns the updated model inventory.

- **URL**: `POST /admin/api/models/refresh`
- **Example Request**:
  ```bash
  curl -s -X POST http://localhost:8082/admin/api/models/refresh
  ```

---

## 3. Inference & Execution Endpoints

### 3.1 OpenAI Chat Completions (`POST /v1/chat/completions`)
Full OpenAI-compatible chat completion endpoint supporting standard JSON responses, Server-Sent Events (SSE) streaming, tool/function calling, and reasoning deltas.

- **URL**: `POST /v1/chat/completions`
- **Headers**: `Content-Type: application/json`

#### Request Payload:
| Field | Type | Description |
| :--- | :--- | :--- |
| `model` | string *(required)* | Exact model ID or Claude alias (e.g. `"openai/gpt-4o"`, `"deepseek/deepseek-chat"`, `"claude-sonnet-4-20250514"`). |
| `messages` | array *(required)* | List of message objects with `role` (`"system"`, `"developer"`, `"user"`, `"assistant"`, `"tool"`) and `content`. |
| `stream` | boolean | Set to `true` for SSE chunk streaming (`data: {...}\n\ndata: [DONE]\n\n`). Default `false`. |
| `tools` | array | List of tool definitions (`{"type": "function", "function": {...}}`). |
| `tool_choice` | string / object | Tool choice policy (`"auto"`, `"none"`, `"required"`, or specific function object). |
| `temperature` | float | Sampling temperature. |
| `max_tokens` / `max_completion_tokens` | integer | Maximum output token limit. |

#### Non-Streaming Example:
```bash
curl -s http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-chat",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

#### Streaming Example (`stream: true`):
```bash
curl -N http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "stream": true
  }'
```
Yields SSE chunks:
```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1727918400,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1727918400,"model":"...","choices":[{"index":0,"delta":{"content":"1, 2, 3, 4, 5"},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1727918400,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

---

### 3.2 Anthropic Messages (`POST /v1/messages`)
Native Anthropic Messages endpoint designed for Anthropic SDKs and Claude Code CLI (`claude`).

- **URL**: `POST /v1/messages`
- **Headers**:
  - `Content-Type: application/json`
  - `anthropic-version: 2023-06-01`
- **Request Payload**:
  ```json
  {
    "model": "claude-3-7-sonnet-20250219",
    "messages": [
      {"role": "user", "content": "Write a Python hello world"}
    ],
    "max_tokens": 1024,
    "stream": false
  }
  ```
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/v1/messages \
    -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
      "model": "claude-3-7-sonnet-20250219",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 1024
    }'
  ```

---

### 3.3 Anthropic Token Counting (`POST /v1/messages/count_tokens`)
Estimates token counts for an Anthropic-formatted payload before making generation calls.

- **URL**: `POST /v1/messages/count_tokens`
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/v1/messages/count_tokens \
    -H "Content-Type: application/json" \
    -d '{
      "model": "claude-3-7-sonnet-20250219",
      "messages": [{"role": "user", "content": "How many tokens?"}]
    }'
  ```

---

### 3.4 OpenAI Responses API (`POST /v1/responses`)
The newer OpenAI Responses standard used by coding agents (such as Muse Code).

- **URL**: `POST /v1/responses`
- **Request Payload**:
  ```json
  {
    "model": "openai/gpt-4o",
    "input": "Refactor this function",
    "stream": true
  }
  ```

---

### 3.5 OpenAI Embeddings (`POST /v1/embeddings`)
Vector embedding generation with automatic mock fallback and multi-key round-robin rotation.

- **URL**: `POST /v1/embeddings`
- **Request Payload**:
  | Field | Type | Description |
  | :--- | :--- | :--- |
  | `input` | string / array of strings *(required)* | Input text(s) to embed. |
  | `model` | string *(required)* | Embedding model name (e.g. `"text-embedding-3-small"`). |
  | `dimensions` | integer *(optional)* | Dimensionality of returned vector. |
- **Example Request**:
  ```bash
  curl -s http://localhost:8082/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "model": "text-embedding-3-small",
      "input": "Search query text",
      "dimensions": 1536
    }'
  ```
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "object": "embedding",
        "index": 0,
        "embedding": [0.0123, -0.0456, 0.0789, ...]
      }
    ],
    "model": "text-embedding-3-small",
    "usage": {
      "prompt_tokens": 4,
      "total_tokens": 4
    }
  }
  ```

---

## 4. Diagnostics & System Endpoints

### 4.1 Server Status (`GET /`)
Returns active default model and provider configuration.
```bash
curl -s http://localhost:8082/
```
Output:
```json
{"status": "ok", "provider": "openai", "model": "openai/gpt-4o"}
```

### 4.2 Health Check (`GET /health`)
Lightweight health probe for orchestrators and container healthchecks.
```bash
curl -s http://localhost:8082/health
```
Output:
```json
{"status": "healthy"}
```

### 4.3 Stop Running Sessions (`POST /stop`)
Immediately cancels pending agent operations and terminates active CLI sessions.
```bash
curl -s -X POST http://localhost:8082/stop
```

---

## 5. Model Naming Convention Guide

When passing a model name in the `"model"` field, FCC supports three formats:

1. **Provider-Qualified Name (`<provider>/<model>` or `<provider>:<model>`)** *(Recommended)*:
   Explicitly tells FCC which provider to route to regardless of defaults.
   - `openai/gpt-4o`
   - `deepseek/deepseek-chat`
   - `groq/llama-3.3-70b-versatile`
   - `open_router/anthropic/claude-3.7-sonnet`
   - `gemini/gemini-2.5-pro`
   - `mistral/codestral-latest`

2. **Direct Upstream Model Name (`<model>`)**:
   Uses the provider currently defined in your settings (`MODEL="provider/model"`).
   - `gpt-4o`
   - `deepseek-chat`

3. **Claude Aliases**:
   Maps standard Claude models to your configured backend model.
   - `claude-3-7-sonnet-20250219`
   - `claude-sonnet-4-20250514`
   - `claude-3-5-haiku-20241022`

---

## 6. Client Integration Examples

### Python: OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8082/v1",
    api_key="none",  # or your FCC_PROXY_TOKEN
)

# 1. Discover available models
models = client.models.list()
for model in models.data:
    print(model.id, "-", getattr(model, "display_name", ""))

# 2. Call chat completions
response = client.chat.completions.create(
    model="deepseek/deepseek-chat",
    messages=[
        {"role": "system", "content": "You are an expert engineer."},
        {"role": "user", "content": "Explain binary search in one sentence."},
    ],
    stream=False,
)
print(response.choices[0].message.content)
```

### Python: Anthropic SDK
```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8082",
    api_key="none",  # or your FCC_PROXY_TOKEN
)

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    messages=[{"role": "user", "content": "Hello Claude!"}],
    max_tokens=100,
)
print(response.content[0].text)
```
