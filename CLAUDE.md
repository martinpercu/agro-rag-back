# CLAUDE.md (backend)

## Que es este repo

Backend de Agroposta: un agente conversacional en rioplatense que responde preguntas sobre la revista "Margenes Agropecuarios" usando RAG (Retrieval-Augmented Generation).

Stack: Python 3.13 + FastAPI + LangGraph + ChromaDB + OpenAI.

## Estructura clave

- `src/api/main.py`: FastAPI app, expone `/chat`, `/chat/stream`, `/compare`, `/export-pdf`
- `src/agent/graph.py`: LangGraph del chat principal (3 nodos: classifier -> retriever -> answerer)
- `src/agent/strategies/`: 6 strategies de retrieval para el comparador + runner async
- `src/ingestion/`: extractor (pdfplumber), chunker, indexer (ChromaDB)
- `src/agent/nodes/answerer.py`: el system prompt rioplatense (gpt-4.1-nano)
- `tests/`: 107 unit + 25 integration tests, golden questions en `tests/golden_questions.json` y `tests/golden_compare_questions.json`

## Comandos utiles

```bash
# Levantar el backend
uv run uvicorn api.main:app --host 127.0.0.1 --port 8002 --app-dir src

# Ingestar un PDF (una vez)
uv run python scripts/ingest_magazine.py data/raw/<archivo>.pdf

# Tests
uv run pytest tests/                    # unit
uv run pytest tests/ --integration      # con OpenAI real

# Report del comparador
uv run python scripts/compare_report.py
```

## Convenciones

- **Model**: gpt-4.1-nano en TODAS las llamadas LLM (answerer, query_rewrite, rerank, multi_query, hyde)
- **Embedding**: text-embedding-3-small
- **Puerto del API**: 8002
- **System prompt**: en `src/agent/nodes/answerer.py:17-69` (NO modificar a menos que sepas que hace)
- **Tests**: pytest con `uv run pytest`. Marcar nuevos tests con `@pytest.mark.skipif("not config.getoption('--integration')")` si requieren API real
- **No commitear**: `data/vector/`, `.venv/`, `__pycache__/`, `.env`, `tmp/`, `.pytest_cache/`

## Si vas a tocar el comparador

Las 6 strategies viven en `src/agent/strategies/`. Cada una implementa `Strategy.retrieve()`. El runner (`runner.py`) las corre en `asyncio.gather`. El answerer es compartido via `agent.nodes.answerer.answer()`.

Para agregar una strategy nueva:
1. Crear `src/agent/strategies/<nombre>.py` con una clase que extienda `Strategy`
2. Agregar a `get_all_strategies()` en `runner.py`
3. Tests en `tests/test_strategies_<nombre>.py`
4. Si tenes LLM calls, usar `call_with_retry` de `llm_retry.py` para manejar rate limits

## Rate limit de OpenAI

El tier default es 200K TPM. Las 6 strategies en paralelo + 6 answerer calls + embeds pueden superar ese limite. Soluciones implementadas:
- `call_with_retry` con backoff exponencial y parsing del delay sugerido por OpenAI
- Report muestra el error real (`❌ llm_failed: ...`) en vez de 0/0 silencioso

Para el golden_compare_questions, los tests unit (sin --integration) no consumen API. Los integration SI.
