# Ollama -> Gemini Proxy (FastAPI)

A minimal FastAPI server that accepts Ollama-compatible endpoints and forwards them to Google Gemini, with runtime-configurable Gemini base URL and API key. Supports SSE streaming and tool-calling, plus a model registry loaded from `.env`.

## Endpoints

- `POST /api/chat`: Compatible with Ollama's `/api/chat`. Supports `stream=true` and `tools`.
- `POST /api/generate`: Compatible with Ollama's `/api/generate`. Supports `stream=true` and `tools`.
- `GET /api/tags`: Lists available models from the `.env` registry (Ollama-style tags listing).

## Configuration

You can configure Gemini target via environment variables and/or request headers. Headers take priority.

Environment variables:

- `GEMINI_BASE_URL` (default `https://generativelanguage.googleapis.com`)
- `GEMINI_API_VERSION` (default `v1beta`)
- `GEMINI_API_KEY` (optional)
- `HOST` (default `0.0.0.0`)
- `PORT` (default `8000`)

Request headers (override env for the request):

- `X-Gemini-Base-Url`
- `X-Gemini-Api-Version`
- `X-Gemini-Api-Key`

### Model Registry via .env

Define a comma-separated list of aliases via `MODELS`, and per-alias settings via `MODEL_<ALIAS>_*`. Example `.env`:

```env
GEMINI_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_API_VERSION=v1beta
GEMINI_API_KEY=global_key_if_any

# Declare two models
MODELS=gemini15pro, flash

# Alias gemini15pro -> actual Gemini model id
MODEL_GEMINI15PRO_ID=gemini-1.5-pro
MODEL_GEMINI15PRO_API_KEY=${GEMINI_API_KEY}
MODEL_GEMINI15PRO_BASE_URL=https://generativelanguage.googleapis.com
MODEL_GEMINI15PRO_API_VERSION=v1beta
MODEL_GEMINI15PRO_DISPLAY_NAME=Gemini 1.5 Pro
MODEL_GEMINI15PRO_DESCRIPTION=High quality, multimodal

# Alias flash -> gemini-1.5-flash with its own key
MODEL_FLASH_ID=gemini-1.5-flash
MODEL_FLASH_API_KEY=another_key
MODEL_FLASH_BASE_URL=https://generativelanguage.googleapis.com
MODEL_FLASH_API_VERSION=v1beta
MODEL_FLASH_DISPLAY_NAME=Gemini 1.5 Flash
MODEL_FLASH_DESCRIPTION=Faster, cheaper
```

The `/api/tags` endpoint will expose these aliases, while each request to `/api/chat` or `/api/generate` using `model: <alias>` will resolve to the underlying `MODEL_<ALIAS>_ID` with corresponding base URL, version, and key.

## Install & Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here  # or override via header per request
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or
python -c "from app.main import run; run()"
```

## Example Requests

`/api/generate`:

```bash
curl -sS http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-1.5-pro",
    "prompt": "Write a haiku about testing",
    "options": {"temperature": 0.7, "max_tokens": 128}
  }'
```

`/api/chat`:

```bash
curl -sS http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-1.5-pro",
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Give me 3 bullet points on FastAPI"}
    ],
    "options": {"temperature": 0.4}
  }'
```

Streaming example:

```bash
curl -N http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini15pro",
    "messages": [
      {"role": "user", "content": "Stream a short sentence token by token"}
    ],
    "stream": true
  }'
```

Override target per request:

```bash
curl -sS http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -H 'X-Gemini-Base-Url: https://generativelanguage.googleapis.com' \
  -H 'X-Gemini-Api-Version: v1beta' \
  -H 'X-Gemini-Api-Key: $GEMINI_API_KEY' \
  -d '{"model": "gemini-1.5-flash", "prompt": "Hello"}'
```

## Notes

- Maps Ollama options `temperature`, `top_p`, `top_k`, `max_tokens` to Gemini `generationConfig`.
- Converts Ollama chat messages to Gemini `contents` with roles: user/system -> `user`, assistant -> `model`; tool outputs map to `functionResponse`.
- Tool-calls in responses are exposed via `tool_calls` to let you invoke your function and then post the result as a `tool` message.
- Error responses from Gemini are forwarded with the same HTTP status.

## Tool Calling

Pass Gemini function declarations via `tools` and return tool results as a follow-up `tool` message.

Declare tools and initiate a call:

```bash
curl -sS http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini15pro",
    "messages": [
      {"role": "user", "content": "What\'s the weather in SF?"}
    ],
    "tools": [
      {
        "function_declarations": [
          {
            "name": "get_weather",
            "description": "Get weather by city name",
            "parameters": {
              "type": "OBJECT",
              "properties": {"city": {"type": "STRING"}},
              "required": ["city"]
            }
          }
        ]
      }
    ]
  }'
```

If the model requests a function call, the response or stream will include `tool_calls` like:

```json
{"tool_calls": [{"type": "function", "name": "get_weather", "args": {"city": "San Francisco"}}]}
```

Then send the tool result back as a `tool` message. Put the function output JSON in `content`, and repeat the `name`:

```bash
curl -sS http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini15pro",
    "messages": [
      {"role": "user", "content": "What\'s the weather in SF?"},
      {"role": "assistant", "content": "calling get_weather(...)"},
      {"role": "tool", "name": "get_weather", "content": "{\"temp\": 21, \"unit\": \"C\"}"}
    ]
  }'
```

## Roadmap

- Add image inputs and multi-part content.
- Add safety settings and tool config helper.
