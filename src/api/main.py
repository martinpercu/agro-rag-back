"""Agroposta API - FastAPI MVP 1.

Endpoints:
- GET  /              -> healthcheck
- GET  /stats         -> metricas del vector store
- POST /compare       -> corre las 6 strategies en paralelo y devuelve el comparison
- POST /compare/stream -> SSE, corre N strategies en paralelo con streaming real
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.nodes.classifier import is_off_topic
from agent.strategies.runner import (
    get_all_strategies,
    get_strategies_by_names,
    run_compare,
    run_compare_stream,
)
from ingestion.indexer import collection_stats

load_dotenv()

ALLOWED_ORIGINS = os.getenv("AGROPOSTA_ALLOWED_ORIGINS", "http://localhost:3002").split(",")
EDITION = os.getenv("AGROPOSTA_EDITION", "2026_05")

app = FastAPI(title="Agroposta API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------

class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] | None = None
    k: int | None = Field(default=6, ge=1, le=20)
    lang: str = "es"


class CompareStreamRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    enabled: list[str] | None = None
    history: list[dict] | None = None
    k: int | None = Field(default=6, ge=1, le=20)
    lang: str = "es"
    sem_bm25: int | None = Field(default=None, ge=1, le=40)  # ancho rama semantica de hybrid
    lex_bm25: int | None = Field(default=None, ge=1, le=40)  # ancho rama BM25 de hybrid
    temperature: float | None = Field(default=None, ge=0, le=1)  # temperatura del answerer


# --------------------------------------------------------------------
# Endpoints basicos
# --------------------------------------------------------------------

@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "agroposta", "edition": EDITION}


@app.get("/stats")
def stats() -> dict:
    return collection_stats()


# --------------------------------------------------------------------
# Comparador RAG: corre las 6 strategies en paralelo
# --------------------------------------------------------------------

@app.post("/compare")
async def compare(req: CompareRequest) -> dict:
    """Corre las 6 strategies de retrieval en paralelo y devuelve todo lado a lado.

    Body: {question, history?, k?, lang?}
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
    result = await run_compare(req.question, req.history, req.k or 6)
    return result.to_dict()


@app.post("/compare/stream")
async def compare_stream(req: CompareStreamRequest) -> StreamingResponse:
    """SSE: corre las strategies habilitadas en paralelo y streamea tokens.

    Body: {question, enabled?, history?, k?, lang?}

    SSE events:
      event: strategy_retrieve  -> {strategy, intent, sources, metrics}
      event: strategy_token     -> {strategy, text}
      event: strategy_done      -> {strategy, answer, sources, metrics}
      event: strategy_error     -> {strategy, error}
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    # Guard: preguntas que no son agropecuarias se rechazan sin gastar nada
    if is_off_topic(req.question):
        msg = (
            "Preguntame sobre costos, márgenes, cultivos o ganadería de la revista Márgenes Agropecuarios."
            if req.lang == "es"
            else "Ask me about costs, margins, crops or livestock from Márgenes Agropecuarios magazine."
        )

        async def _reject():
            names = req.enabled or [s.name for s in get_all_strategies()]
            for name in names:
                yield f"event: strategy_error\ndata: {json.dumps({'strategy': name, 'error': msg})}\n\n"

        return StreamingResponse(_reject(), media_type="text/event-stream")

    all_strategies = get_all_strategies()
    hybrid_kwargs: dict = {}
    if req.sem_bm25 is not None:
        hybrid_kwargs["chroma_top_k"] = req.sem_bm25
    if req.lex_bm25 is not None:
        hybrid_kwargs["bm25_top_k"] = req.lex_bm25
    if req.enabled:
        strategies = get_strategies_by_names(req.enabled, **hybrid_kwargs)
    else:
        names = [s.name for s in all_strategies]
        strategies = get_strategies_by_names(names, **hybrid_kwargs)

    async def _format_events():
        async for msg in run_compare_stream(
            req.question, req.history, req.k or 6, strategies, temperature=req.temperature
        ):
            name = msg["strategy"]
            typ = msg["type"]
            data = msg["data"]

            if typ == "token":
                yield f"event: strategy_token\ndata: {json.dumps({'strategy': name, 'text': data})}\n\n"
            elif typ == "retrieve_done":
                payload: dict = {"strategy": name, **data}
                yield f"event: strategy_retrieve\ndata: {json.dumps(payload)}\n\n"
            elif typ == "done":
                payload = {"strategy": name, **data}
                yield f"event: strategy_done\ndata: {json.dumps(payload)}\n\n"
            elif typ == "error":
                yield f"event: strategy_error\ndata: {json.dumps({'strategy': name, 'error': data})}\n\n"

    return StreamingResponse(_format_events(), media_type="text/event-stream")



