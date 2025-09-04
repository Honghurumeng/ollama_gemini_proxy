# Repository Guidelines

## Project Structure & Module Organization
- `app/`: FastAPI service code
  - `main.py`: app factory, CORS, logging middleware, `run()`
  - `routers.py`: API routes (`/api/chat`, `/api/generate`, `/api/tags`, `/api/show`, `/api/version`, `/v1/chat/completions`, `/v1/models`)
  - `gemini_client.py`: HTTPX client (streaming, debug saving via `DEBUG_SAVE_RESPONSES` → `debug_responses/`)
  - `config.py`: env-based settings, model registry (`MODELS`, `MODEL_<ALIAS>_*`), header overrides
  - `models.py`: Pydantic schemas (Ollama ↔︎ Gemini)
  - `logger.py`: JSON/plain logging; `LOG_LEVEL`, `LOG_JSON`, `LOG_MAX_BODY`
- Config: `.env.example` (copy to `.env`).
- Tests: place future tests under `tests/`.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run (dev)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Or
python -c "from app.main import run; run()"

# Quick checks
curl -s http://localhost:8000/api/tags | jq '.models[0].name'
curl -s -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini25pro","messages":[{"role":"user","content":"ping"}]}'
```

## Coding Style & Naming Conventions
- Python/PEP 8; 4-space indentation; type hints required.
- Names: `snake_case` for modules/functions, `CapWords` for classes, `UPPER_SNAKE` constants.
- Route handlers follow `api_*` naming; keep response shapes Ollama/OpenAI compatible.
- Use `get_logger()`; never `print()`; avoid logging secrets (helpers already mask/truncate).

## Testing Guidelines
- Framework: `pytest` (add under `tests/`), files named `test_*.py`.
- Prioritize: `routers.py` (endpoints and streaming parsers), `config.py` (override precedence).
- Example: use `fastapi.testclient.TestClient` to exercise `/api/*` and `/v1/*` routes.
- Run: `pytest -q` (once a suite exists). Aim for coverage on critical paths.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- PRs include: clear description, linked issues, test plan (curl examples), config notes (.env keys), and updates to `README.md`/`CLAUDE.md` if APIs or env vars change.
- Validate locally before opening: server starts, example curls succeed, no secrets in diffs.

## Security & Configuration Tips
- Do not commit secrets; use `.env` or request headers (`X-Gemini-*`).
- Config precedence: headers > per-model env > global env > defaults.
- Leave `DEBUG_SAVE_RESPONSES` off in production; avoid committing `debug_responses/`.
