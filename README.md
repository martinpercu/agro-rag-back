# Agroposta Backend

Backend de Agroposta: agente conversacional en rioplatense que responde preguntas sobre la revista Margenes Agropecuarios usando RAG.

## Stack

- **Python 3.13** + uv
- **FastAPI** + uvicorn
- **LangGraph 1.0** (3 nodos: classifier, retriever, answerer)
- **ChromaDB** (vector store persistente, local)
- **OpenAI** (gpt-4.1-nano + text-embedding-3-small)
- **fpdf2** (export de PDF)

## Estructura

```
agro-back/
├── src/
│   ├── agent/                LangGraph + las 6 strategies del comparador
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes/            classifier, retriever, answerer
│   │   └── strategies/       6 strategies + runner async
│   ├── ingestion/            extractor (pdfplumber), chunker, indexer
│   ├── export/               pdf_generator (fpdf2)
│   ├── api/                  FastAPI app
│   └── schemas.py            Pydantic models
├── tests/                   107 unit + integration tests
├── scripts/                  ingesta, compare report, etc
├── data/
│   ├── raw/                  PDFs fuente (ej margenes_2026_05.pdf)
│   └── vector/               ChromaDB (gitignored, regenerable)
├── pyproject.toml
└── uv.lock
```

## Setup

```bash
# 1. Crear venv e instalar deps
uv sync

# 2. Configurar API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Ingestar el PDF (una vez por edicion)
uv run python scripts/ingest_magazine.py data/raw/margenes_2026_05.pdf

# 4. Levantar el API
uv run uvicorn api.main:app --host 127.0.0.1 --port 8002 --app-dir src
```

## Tests

```bash
# Solo unit (no gasta API)
uv run pytest tests/

# Con integration (gasta API)
uv run pytest tests/ --integration
```

## Endpoints

- `GET /` - healthcheck
- `GET /stats` - metricas del vector store
- `POST /chat` - una pregunta, devuelve {answer, sources, intent}
- `POST /chat/stream` - SSE, streamea la respuesta
- `POST /compare` - corre las 6 strategies en paralelo, devuelve metricas
- `POST /export-pdf` - genera PDF de la conversacion
- `GET /sessions/{id}` - lee mensajes de una sesion
- `DELETE /sessions/{id}` - borra sesion

## Las 6 strategies del comparador

| # | Nombre | Que hace |
|---|---|---|
| 1 | baseline | Busqueda semantica + filtro por intent |
| 2 | hybrid | BM25 (lexico) + semantica, merge con RRF |
| 3 | rerank | Top-20 semantico + LLM rerank → top-6 |
| 4 | query_rewrite | LLM reescribe la query con history → semantica |
| 5 | multi_query | LLM genera 3 reformulaciones + RRF |
| 6 | hyde | LLM genera respuesta hipotetica → embed → search |

El answerer es el MISMO para las 6 (system prompt rioplatense, gpt-4.1-nano). Solo cambia el retrieval.

## Ver el CLAUDE.md

`cat CLAUDE.md` para contexto adicional sobre el proyecto.
