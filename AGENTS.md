# AGENTS.md (backend)

## What is this repo

Backend for Agroposta: a rioplatense conversational agent that answers questions about "Margenes Agropecuarios" magazine using RAG.

Stack: Python 3.13 + FastAPI + LangGraph + ChromaDB/Pinecone + OpenAI gpt-4.1-nano + Supabase Auth + Railway Postgres.

**State 2026-09-04:** single DB 768 `text-embedding-nomic-embed-text-v1.5` in Chroma local (107 docs, `VECTOR_STORE=chroma`) and `text-embedding-3-small__d768` in Pinecone `agro-vectorstore` 768 cosine serverless (107 docs, `VECTOR_STORE=pinecone`). 5-node LangGraph `classifier→plan_intent→field_collector→retriever→answerer` fully traced with Langfuse local-only (lab vs user separation). All feature branches merged to `main` (`ddde9a6`).

**State 2026-09-15 (satellite, unmerged):** `src/satellite/` module (Sentinel-2 NDVI, fully separate from RAG) on stacked branches `feat/sat-location-contract` → `feat/sat-sentinel-client` → `feat/sat-ndvi-endpoint` (each contains its parent). Live-verified against Copernicus with real OAuth client. See Satellite section below.

## Stack local vs OpenAI (provider factory)

All clients via `src/agent/llm.py` reading env vars:

| Var | Default | Usage |
|---|---|---|
| `AGROPOSTA_LLM_BASE_URL` | (empty → OpenAI) | Chat LLM base URL (e.g. LM Studio `http://192.168.12.215:1234/v1`) |
| `AGROPOSTA_LLM_MODEL` | `gpt-4.1-nano` | Chat model (e.g. `qwen/qwen3.6-35b-a3b`) |
| `AGROPOSTA_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings model (e.g. `text-embedding-nomic-embed-text-v1.5`) |
| `AGROPOSTA_EMBEDDING_DIMS` | (empty → native) | Matryoshka dims for `text-embedding-3-small` (e.g. `768`) |
| `AGROPOSTA_RERANK_URL` | `http://192.168.12.215:8001/v1/rerank` | Cross-encoder rerank service (Jina-compatible) |
| `OPENAI_API_KEY` | — | Key. Any string works for local server (e.g. `lm-studio`) |
| `VECTOR_STORE` | `chroma` | `chroma` (local file `data/vector/`) or `pinecone` (serverless) |
| `PINECONE_API_KEY` / `PINECONE_INDEX` | — | Only if `VECTOR_STORE=pinecone` |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` / `SUPABASE_JWT_AUD` | — | Supabase Auth (see Auth section) |
| `DATABASE_URL` | — | Railway Postgres (`postgresql+psycopg://...@altaria.proxy.rlwy.net:42134/railway`) |
| `SENTINEL_CLIENT_ID` / `SENTINEL_CLIENT_SECRET` | — | Copernicus OAuth client (satellite NDVI, see Satellite section; never commit) |
| `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | Local Langfuse `http://localhost:3003` (never in `.env`, only `.env.local`) |

- `collection_name()` derives collection from embedding model + dims: default keeps `margenes_agropecuarios` (1536), with `AGROPOSTA_EMBEDDING_DIMS=768` → `margenes_agropecuarios__d768`. Other models → `margenes_agropecuarios__<modelo>`.
- **Don't use `langchain_openai.OpenAIEmbeddings`**: tokenizes and sends token IDs, LM Studio rejects. Use `get_embeddings()` (own adapter over SDK).
- Profile local is in `.env.local` (load with `set -a; source .env.local; set +a`; `load_dotenv` only reads `.env`).
- **Switch local:on/off**: `source .env.local` (local LLM `qwen` + nomic + `VECTOR_STORE=chroma`) vs `source .env` (OpenAI `gpt-4.1-nano` + `text-embedding-3-small__d768` + `VECTOR_STORE=pinecone`). Check `GET /` → `vector_store` + `vector_health`.
- **VectorStore abstraction** in `src/ingestion/vector_store.py`: `VECTOR_STORE=chroma` delegates to `indexer.py` (Chroma file), `pinecone` delegates to Pinecone Serverless (768d `cosine`, free tier). Strategies import `indexer.search` which auto-delegates.

## Key structure

- `src/api/main.py`: FastAPI app — `/`, `/stats`, `/chat` (graph, `user:chat`), `/compare`, `/compare/stream` (lab `lab:compare_stream` vs product `user:compare_stream`), `/plan/parse`, `/export-pdf`, `/me`, `/investigations`, `/plans`, `/editions`, `/satellite/ndvi` + `/satellite/health` (Sentinel-2, open, 24h cache)
- `src/agent/graph.py`: LangGraph `make_graph()` with 5 nodes: `classifier` → `plan_intent` → `field_collector` → `retriever` → `answerer` (`StateGraph(AgentState)`)
- `src/agent/nodes/`: `classifier.py` (rule-based intent), `plan_intent.py` (`is_plan_intent`), `field_collector.py` (`extract_divisions`), `retriever.py` (`search` with `INTENT_TO_SECTION`), `answerer.py` (rioplatense system prompt `gpt-4.1-nano`, `answer()` + `answerer_node()` + streaming)
- `src/observability.py`: canonical Langfuse helper SDK 2.80 — `get_langfuse()`, `start_as_current_observation(..., session_id, user_id, tags, model, usage_details)`, `update_current_observation()`, `flush()` (no OTel)
- `src/instrumentation.py`: deprecated shim re-exporting `observability` (kept for `import instrumentation` compat)
- `src/agent/strategies/`: 6 retrieval strategies + `rerank_ce` extra + `runner.py` (`asyncio.gather`, `run_compare_stream` SSE)
- `src/agent/rerank_client.py`: cross-encoder HTTP client
- `src/ingestion/`: extractor (pdfplumber), chunker, indexer (ChromaDB)
- `src/db/`: SQLAlchemy models (`User`, `Investigation`, `Plan`, `Session`), `session.py` (`get_db`)
- `src/api/auth.py`: Supabase JWT verification (JWKS ES256, 10min cache)
- `src/satellite/`: Sentinel-2 NDVI module, fully separate from RAG (no LLM, no vector store) — `geo.py` (location contract), `sentinel.py` (OAuth + Statistical API client), `cache.py` (TTLCache)
- `scripts/satellite_demo.py`: live NDVI demo (needs `SENTINEL_*` creds, exits 2 without them)
- `tests/`: unit + integration, golden questions `tests/golden_questions.json` / `tests/golden_compare_questions.json`, satellite `tests/test_location_geo.py` / `tests/test_satellite_client.py` / `tests/test_api_satellite.py` (mocked HTTP, no quota spend)

## Useful commands — switch local:on/off + vectorstore

```bash
# Local single DB 768 (nomic via LM Studio + rerank :8001)
set -a; source .env.local; set +a  # VECTOR_STORE=chroma, nomic 768
uv run uvicorn api.main:app --host 127.0.0.1 --port 8002 --app-dir src
curl http://127.0.0.1:8002/ | jq .vector_health  # → chroma nomic 768

# Prod Pinecone 768 (OpenAI, without Mac mini) + Langfuse local
set -a; source .env; set +a
export LANGFUSE_HOST=http://localhost:3003
export LANGFUSE_PUBLIC_KEY=pk-lf-1a76f15c-9fa5-41fd-a58f-ed4755633990
export LANGFUSE_SECRET_KEY=sk-lf-8557a1d4-ac9b-47bc-ace9-997df892edbc
uv run uvicorn api.main:app --host 127.0.0.1 --port 8002 --app-dir src  # pinecone 768, total 107
curl http://127.0.0.1:8002/ | jq .vector_health  # → pinecone agro-vectorstore 768
curl http://127.0.0.1:8002/stats | jq .total       # → 107

# Ingest a PDF (once) — respects VECTOR_STORE
uv run python scripts/ingest_magazine.py data/raw/<file>.pdf

# Tests
uv run pytest tests/                    # unit (160 passed, 33 skipped without --integration)
uv run pytest tests/ --integration      # with real OpenAI

# Compare report
uv run python scripts/compare_report.py

# Bakeoff embeddings (golden questions x baseline/hybrid, no LLM)
uv run python scripts/embedding_bakeoff.py --models <comma-separated>

# Langfuse local-only (traces nodes) — never to prod
docker compose -f docker-compose.langfuse.yml up -d  # UI http://localhost:3003, DB 5434:5432, Redis 6380, Minio 9090
# then: set -a; source .env.local; set +a  # LANGFUSE_HOST=http://localhost:3003
uv run uvicorn api.main:app --port 8002 --app-dir src  # view at http://localhost:3003/project/cmtm1hco50006jt0775kjzrvg
```

## Observability — Langfuse local-only (never prod)

- **Stack:** `langfuse:3003:3000` + `langfuse-worker` + `db:5434:5432` Postgres + `clickhouse:8123` + `redis:6380` + `minio:9090/9091` (`docker-compose.langfuse.yml`, volumes `pg_data/ch_data` persistence). Health `GET http://localhost:3003/api/public/health` → `3.225.7` (v3).
- **Activation:** only if `LANGFUSE_HOST=http://localhost:3003` in `.env.local` (not in `.env` → Railway doesn't trace). Uses `src/observability.py:1` manual SDK 2.80, **no OTel** (avoids `Connection reset by peer` on `/api/public/otel/v1/traces`). `src/instrumentation.py` is no-op shim.
- **Tracer separation by nature (2026-09-04):**
  - `POST /chat` → `user:chat` `session_id=userId` (`_extract_session_id()` reads `Authorization: Bearer <jwt>.sub` or `X-User-Id`, fallback `anon`) + 5 child spans `classifier` (`span`), `plan_intent` (`span`), `field_collector` (`span`), `retriever` (`retriever`), `answerer` (`generation` with `model=gpt-4.1-nano` + `usageDetails:{input,output}` + `cost`). Groups by user in Langfuse `Users`/`Sessions` (tester vs alice vs bob distinct).
  - `POST /compare/stream` → `user:compare_stream` `session_id=tester` `tags:["user"]` when `enabled==["baseline"]` (product `/` streaming, keeps streaming), else `lab:compare_stream` `session_id="lab"` `tags:["lab"]` (research `/dev` multi-strategy, single lab user). Both wrap SSE streaming with `start_as_current_observation` + `flush()` after stream; product keeps streaming UX.
  - `POST /plan/parse` → `plan_parse` `session_id=userId` (lightweight, no LLM, 1 span).
- **Helper:** `from observability import get_langfuse, start_as_current_observation, update_current_observation, flush` + `from langfuse import propagate_attributes`. Each node wraps `with lf.start_as_current_observation(name="classifier", as_type="span", input={question,history})` + `lf.update_current_span(output={intent})`. See `src/agent/nodes/classifier.py:304` etc.
- **Verification:**
  ```bash
  curl -X POST http://127.0.0.1:8002/chat -H "X-User-Id: tester" -H "Content-Type: application/json" -d '{"question":"costo maiz"}' | jq
  curl -X POST http://127.0.0.1:8002/compare/stream -H "X-User-Id: tester" -d '{"question":"costo maiz","enabled":["baseline"]}' --no-buffer | head
  # Clickhouse:
  docker exec agro-rag-back-clickhouse-1 clickhouse-client --query "SELECT name, session_id, tags FROM traces ORDER BY timestamp DESC LIMIT 5"
  # → user:compare_stream tester ['user'], lab:compare_stream lab ['lab'], user:chat tester
  ```

## Auth + DB — Supabase + Railway Postgres (Fase 0)

- **Supabase Auth:** `SUPABASE_URL=https://ujytfzskyizupdvuxewq.supabase.co` `SUPABASE_ANON_KEY=sb_publishable_...` `SUPABASE_JWT_AUD=authenticated`. `src/api/auth.py:1` verifies ES256 via `JWKS https://.../auth/v1/.well-known/jwks.json`, cache 10min. Frontend `agro-rag-front/app/lib/supabase.ts` `supabase.auth.getSession()` sends `Authorization: Bearer <access_token>` to `POST /compare/stream`, `POST /plan/parse`, `POST /investigations` (`app/(dashboard)/page.tsx:142`).
- **DB:** `DATABASE_URL=postgresql+psycopg://postgres:...@altaria.proxy.rlwy.net:42134/railway` (Railway). Models `db/models.py` `User(supabase_user_id)`, `Investigation`, `Plan`, `Session`. `_ensure_user()` in `src/api/main.py:322` get-or-create via `sub` (UUID or `uuid5` for non-UUID `dev-user`).
- **Keepalive:** Supabase pauses if no auth activity. Keep alive via `POST /auth/v1/signup` (e.g. `keepalive-...@gmail.com`) or any `supabase.auth.getSession()` from front. Last keepalive `98445c8d-dd3d-45a9-9383-f92239ab60e5` via `node /tmp/supabase_keepalive3.cjs`.

## Graph — 5 nodes

`src/agent/graph.py:12` `make_graph()`:
```
START → classifier → plan_intent → field_collector → retriever → answerer → END
```
- `classifier`: rule-based `_classify()` → `Intent` (`costos|mercado|proyecciones|tecnologia|ganaderia|siembras|general`), also `is_off_topic()`
- `plan_intent`: `is_plan_intent(question, history)` detects `plan de siembra` / `ha` + crop
- `field_collector`: `extract_divisions(question, history)` → `[{hectares:"120", cultivo:"soja"}]`, location `{}` (open for DI-5)
- `retriever`: `INTENT_TO_SECTION` → `search(question, k=6, where={"seccion":{"$in":...}})` via `ingestion/indexer.py` / `vector_store.py`
- `answerer`: `answer(question, items)` + `answerer_node(state)` wraps `get_chat_client()` / `get_async_chat_client()` `llm_model()` `TEMPERATURE 0.2` `MAX_TOKENS 900` `SYSTEM_PROMPT` rioplatense, returns `{answer, sources, input_tokens, output_tokens}`

`AgentState` (`src/agent/state.py:1`): `question`, `intent`, `retrieved`, `answer`, `sources`, `plan_intent`, `divisions`, `location`, `history`.

## Satellite NDVI — Sentinel-2 via Copernicus (separate from RAG, 2026-09-15)

Decision: Sentinel-2 only, Auravant discarded. NDVI = reflectance `(B08-B04)/(B08+B04)`, never a diagnosis (red = possible stress OR bare soil/clouds — UI carries the disclaimer).

- **Location contract** (`src/satellite/geo.py`, stdlib only): every input normalizes to `{polygon (GeoJSON EPSG:4326), centroid, area_ha, origin:{mode}, label?}`. Accepted: `{}` legacy, `{lat,lng,ha}` point+radius (1–20 ha → 32-vertex circle), `{vertices:[{lat,lng}|[lng,lat]]}` (auto-closes), `{bbox:{lat_min,...}}`, `{polygon}` or raw GeoJSON Polygon. Validation: Argentina bbox approx, ≤500 vertices, ≤50.000 ha. `POST /investigations` + `POST /plans` normalize (422 on invalid); `/plan/parse` still returns `{}`.
- **Client** (`src/satellite/sentinel.py`, sync `httpx` like `rerank_client.py`): OAuth2 `client_credentials` against `identity.dataspace.copernicus.eu/.../CDSE/.../token` (token cached ~60s before expiry, single retry on 401) + `POST {SH_BASE}/api/v1/statistics` with NDVI evalscript (`B04/B08+dataMask`), Sentinel-2 L2A, `maxCloudCoverage:10`, P5D default (P1D optional). Empty (fully cloudy) buckets kept with `ndvi_mean: null`.
- **Gotcha — resolution units:** Statistical API reads `resx/resy` in geometry CRS units → EPSG:4326 means **degrees** (`resolution_m / 111320`, ~0.0001° for 10m). Passing meters yields 1 giant pixel (`sample_count: 1`). Caught live 2026-09-15.
- **Endpoints** (`src/api/main.py`): `GET /satellite/health` (token check, no PU spend) + `POST /satellite/ndvi {location|polygon, date_from?, date_to? (default last 90d), aggregation? P5D|P1D}` → `{series, polygon, centroid, area_ha, origin, aggregation, cached}`. Open (like `/plan/parse`); quota guard: 24h `TTLCache` keyed by polygon+dates+agg, max range 365d. Errors: 422 bad location/dates, 429 quota, 500 satellite misconfigured, 502 upstream.
- **OAuth setup (martin does this, one per rotated secret):** login `dataspace.copernicus.eu` → profile → Sentinel Hub → User Settings → OAuth clients → Create (NOT a SPA — secret stays backend-only) → `SENTINEL_CLIENT_ID/SECRET` into `.env` (gitignored). Code reads `SENTINEL_*`; a bare `CLIENT_ID=` will silently miss (fail-fast exits 2 in demo).
- **Free quota:** 10.000 req + 10.000 PU/month, 300/min (resets 1st). One P5D series ≈ 1 request.
- **Verification:**
  ```bash
  uv run python scripts/satellite_demo.py            # live, needs creds
  curl http://127.0.0.1:8002/satellite/health
  curl -X POST http://127.0.0.1:8002/satellite/ndvi -H 'Content-Type: application/json' -d '{"location":{"lat":-34.5,"lng":-62.0,"ha":5}}'
  ```

## Conventions

- **Model:** `gpt-4.1-nano` in ALL LLM calls (answerer, query_rewrite, rerank, multi_query, hyde) — unless `AGROPOSTA_LLM_MODEL` overrides
- **Embedding:** `text-embedding-3-small` default — unless `AGROPOSTA_EMBEDDING_MODEL` overrides
- **Port:** `8002`
- **System prompt:** in `src/agent/nodes/answerer.py:22` (don't modify unless you know what it does)
- **Tests:** `pytest` with `uv run pytest`. Mark new tests with `@pytest.mark.skipif("not config.getoption('--integration')")` if they need real API
- **Don't commit:** `data/vector/`, `.venv/`, `__pycache__/`, `.env`, `.env.local`, `tmp/`, `.pytest_cache/`

## Branch workflow (mandatory since 2026-09-03)

Never to `main` directly. Baby steps in independent branch per repo:

```bash
git checkout -b feat/<name> && git push -u origin feat/<name>
# ... small commits ...
git status; git diff; git log --oneline -5
git add <files> && git commit -m "feat: ..." && git push
# PR → merge to main done by martin in GitHub (Railway deploys)
```

Branches front and back **DO NOT go in parallel by default**. Each repo handles its own branch if the change is only there. Only when the session requires implementing from both sides (e.g. API contract change) create branches with same name in both and coordinate from root.

## If you touch the comparator

Strategies in `src/agent/strategies/`. Each implements `Strategy.retrieve()`. Runner (`runner.py`) runs them with `asyncio.gather`. Answerer is shared via `agent.nodes.answerer.answer()`.

- `get_all_strategies()`: 6 default (`baseline`, `hybrid`, `rerank`, `query_rewrite`, `multi_query`, `hyde`)
- `get_extra_strategies()`: outside default — only `rerank_ce` (cross-encoder, no LLM)
- `get_strategies_by_names(names)`: default + extra, for `/compare/stream` with `enabled`

To add a new strategy:
1. Create `src/agent/strategies/<name>.py` extending `Strategy`
2. Add to `get_all_strategies()` or `get_extra_strategies()` in `runner.py`
3. Tests in `tests/test_strategies_<name>.py`
4. If LLM calls, use `call_with_retry` from `llm_retry.py`

**ChromaDB not thread-safe:** strategies run in parallel (`asyncio.to_thread`). `ingestion/indexer.py` uses singleton client per path + `threading.Lock`. Don't create `PersistentClient` per call.

**BM25 Cache** (`hybrid.py`): keyed by `collection:count` — rebuilds only if ingested to another collection with same count.

## OpenAI rate limit

Default tier 200K TPM. 6 strategies in parallel + 6 answerer calls + embeds can exceed. Implemented:
- `call_with_retry` with exponential backoff and parsing OpenAI suggested delay
- Report shows real error (`❌ llm_failed: ...`) instead of silent 0/0

For `golden_compare_questions`, unit tests (without --integration) don't consume API. Integration ones do.
