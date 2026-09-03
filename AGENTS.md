# AGENTS.md (backend)

## Que es este repo

Backend de Agroposta: un agente conversacional en rioplatense que responde preguntas sobre la revista "Margenes Agropecuarios" usando RAG (Retrieval-Augmented Generation).

Stack: Python 3.13 + FastAPI + LangGraph + ChromaDB/Pinecone + OpenAI.

**Estado 2026-08-30:** single DB 768 `text-embedding-nomic-embed-text-v1.5` en Chroma local (107 docs, `VECTOR_STORE=chroma`), y `text-embedding-3-small__d768` en Pinecone `agro-vectorstore` 768 cosine serverless (107 docs, `VECTOR_STORE=pinecone`). Ver `agroposta-plans/README.md` para plan 2 fases.

## Stack local vs OpenAI (provider factory)

Todos los clientes se crean via `src/agent/llm.py`, que lee env vars:

| Var | Default | Uso |
|---|---|---|
| `AGROPOSTA_LLM_BASE_URL` | (vacío → OpenAI) | Base URL del chat LLM (ej: LM Studio `http://192.168.12.215:1234/v1`) |
| `AGROPOSTA_LLM_MODEL` | `gpt-4.1-nano` | Modelo del chat (ej: `qwen/qwen3.6-35b-a3b`) |
| `AGROPOSTA_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings (ej: `text-embedding-nomic-embed-text-v1.5`) |
| `AGROPOSTA_EMBEDDING_DIMS` | (vacío → native) | Dims Matryoshka para `text-embedding-3-small` (ej: `768`) |
| `AGROPOSTA_RERANK_URL` | `http://192.168.12.215:8001/v1/rerank` | Servicio de cross-encoder rerank (Jina-compatible) |
| `OPENAI_API_KEY` | — | Key. Con un servidor local cualquiera sirve (ej: `lm-studio`) |
| `VECTOR_STORE` | `chroma` | `chroma` (local file `data/vector/`) o `pinecone` (serverless) |
| `PINECONE_API_KEY` / `PINECONE_INDEX` | — | Solo si `VECTOR_STORE=pinecone` |

- `collection_name()` deriva el nombre de la coleccion del embedding model + dims: default sin dims conserva `margenes_agropecuarios` (1536), con `AGROPOSTA_EMBEDDING_DIMS=768` → `margenes_agropecuarios__d768`. Otros modelos → `margenes_agropecuarios__<modelo>`.
- **No usar `langchain_openai.OpenAIEmbeddings`**: tokeniza y manda token IDs, que LM Studio rechaza. Usar `get_embeddings()` (adapter propio sobre el SDK).
- El profile local esta en `.env.local` (cargar con `set -a; source .env.local; set +a`; `load_dotenv` solo lee `.env`).
- **Switch local:on/off**: `source .env.local` (local LLM `qwen` + bge-m3 + `VECTOR_STORE=chroma`) vs `source .env` (OpenAI `gpt-4.1-nano` + `text-embedding-3-small__d768` + `VECTOR_STORE=chroma|pinecone`). Ver `GET /` → `vector_store` + `vector_health` para verificar.
- **VectorStore abstraction** en `src/ingestion/vector_store.py`: `VECTOR_STORE=chroma` delega a `indexer.py` (Chroma file), `pinecone` delega a Pinecone Serverless (768d `cosine`, free tier). Las strategies importan `indexer.search` que ya delega automaticamente.

## Estructura clave

- `src/api/main.py`: FastAPI app, expone `/chat`, `/chat/stream`, `/compare`, `/export-pdf`
- `src/agent/graph.py`: LangGraph del chat principal (3 nodos: classifier -> retriever -> answerer)
- `src/agent/strategies/`: 6 strategies de retrieval para el comparador + `rerank_ce` (extra) + runner async
- `src/agent/rerank_client.py`: cliente HTTP del servicio de cross-encoder rerank
- `src/ingestion/`: extractor (pdfplumber), chunker, indexer (ChromaDB)
- `src/agent/nodes/answerer.py`: el system prompt rioplatense (gpt-4.1-nano)
- `tests/`: unit + integration tests, golden questions en `tests/golden_questions.json` y `tests/golden_compare_questions.json`

## Comandos utiles — switch local:on/off + vectorstore

```bash
# Local single DB 768 (nomic via LM Studio + rerank :8001)
set -a; source .env.local; set +a  # VECTOR_STORE=chroma, nomic 768
uv run uvicorn api.main:app --host 127.0.0.1 --port 8002 --app-dir src
curl http://127.0.0.1:8002/ | jq .vector_health  # → chroma nomic 768

# Prod Pinecone 768 (OpenAI, sin Mac mini)
set -a; source .env; set +a  # VECTOR_STORE=pinecone, PINECONE_INDEX=agro-vectorstore, DIMS=768
uv run uvicorn api.main:app --host 127.0.0.1 --port 8002 --app-dir src
curl http://127.0.0.1:8002/ | jq .vector_health  # → pinecone agro-vectorstore 768
curl http://127.0.0.1:8002/stats | jq .total       # → 107

# Ingestar un PDF (una vez) — respeta VECTOR_STORE
uv run python scripts/ingest_magazine.py data/raw/<archivo>.pdf

# Tests
uv run pytest tests/                    # unit
uv run pytest tests/ --integration      # con OpenAI real

# Report del comparador
uv run python scripts/compare_report.py

# Bakeoff de embeddings (golden questions x baseline/hybrid, sin LLM)
uv run python scripts/embedding_bakeoff.py --models <modelos separados por coma>
```

## Convenciones

- **Model**: gpt-4.1-nano en TODAS las llamadas LLM (answerer, query_rewrite, rerank, multi_query, hyde) — salvo que `AGROPOSTA_LLM_MODEL` lo sobreescriba
- **Embedding**: text-embedding-3-small (default) — salvo que `AGROPOSTA_EMBEDDING_MODEL` lo sobreescriba
- **Puerto del API**: 8002
- **System prompt**: en `src/agent/nodes/answerer.py:17-69` (NO modificar a menos que sepas que hace)
- **Tests**: pytest con `uv run pytest`. Marcar nuevos tests con `@pytest.mark.skipif("not config.getoption('--integration')")` si requieren API real
- **No commitear**: `data/vector/`, `.venv/`, `__pycache__/`, `.env`, `.env.local`, `tmp/`, `.pytest_cache/`

## Workflow de ramas (obligatorio desde 2026-09-03)

Nunca a `main` directo. Baby steps en rama paralela back+front con mismo nombre:

```bash
git checkout -b feat/<nombre> && git push -u origin feat/<nombre>
# ... commits chicos ...
git status; git diff; git log --oneline -5
git add <archivos> && git commit -m "feat: ..." && git push
# PR → merge a main lo hace martin en GitHub (Railway deploya)
```

## Si vas a tocar el comparador

Las strategies viven en `src/agent/strategies/`. Cada una implementa `Strategy.retrieve()`. El runner (`runner.py`) las corre en `asyncio.gather`. El answerer es compartido via `agent.nodes.answerer.answer()`.

- `get_all_strategies()`: las 6 del comparador default (baseline, hybrid, rerank, query_rewrite, multi_query, hyde)
- `get_extra_strategies()`: fuera del default — hoy solo `rerank_ce` (cross-encoder, no LLM)
- `get_strategies_by_names(names)`: default + extra, para `/compare/stream` con `enabled`

Para agregar una strategy nueva:
1. Crear `src/agent/strategies/<nombre>.py` con una clase que extienda `Strategy`
2. Agregar a `get_all_strategies()` o `get_extra_strategies()` en `runner.py`
3. Tests en `tests/test_strategies_<nombre>.py`
4. Si tenes LLM calls, usar `call_with_retry` de `llm_retry.py` para manejar rate limits

**ChromaDB no es thread-safe**: las strategies corren en paralelo (`asyncio.to_thread`). `ingestion/indexer.py` usa un client singleton por path + `threading.Lock` en las operaciones. No crear `PersistentClient` por llamada (rompe con tenant errors).

**Cache BM25** (`hybrid.py`): keyed por `coleccion:count` — si ingestas a otra coleccion con la misma cantidad de chunks, se reconstruye solo.

## Rate limit de OpenAI

El tier default es 200K TPM. Las 6 strategies en paralelo + 6 answerer calls + embeds pueden superar ese limite. Soluciones implementadas:
- `call_with_retry` con backoff exponencial y parsing del delay sugerido por OpenAI
- Report muestra el error real (`❌ llm_failed: ...`) en vez de 0/0 silencioso

Para el golden_compare_questions, los tests unit (sin --integration) no consumen API. Los integration SI.
