"""Agroposta API - FastAPI MVP 1.

Endpoints:
- GET  /              -> healthcheck
- GET  /stats         -> metricas del vector store
- POST /chat          -> pregunta puntual, devuelve {answer, sources, intent}
- POST /chat/stream   -> mismo, pero SSE token-por-token
- POST /compare       -> corre las 6 strategies en paralelo y devuelve el comparison
- POST /export-pdf    -> genera PDF a partir de la conversacion en memoria
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from agent.graph import graph
from agent.strategies.runner import run_compare
from export.pdf_generator import render_conversation_pdf
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

# Memoria volatil: en MVP 1 las sesiones viven en este dict y se pierden
# al reiniciar el servidor. Suficiente para la demo.
SESSIONS: dict[str, list[dict]] = {}


# --------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------

Role = Literal["user", "assistant"]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = None


class SourceItem(BaseModel):
    pagina: int
    seccion: str
    cultivo: str | None = None
    campana: str | None = None
    tipo: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    sources: list[SourceItem]


class MessageItem(BaseModel):
    role: Role
    content: str
    sources: list[SourceItem] | None = None


class ExportPdfRequest(BaseModel):
    session_id: str


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] | None = None
    k: int | None = Field(default=6, ge=1, le=20)


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
# Chat
# --------------------------------------------------------------------

def _run_agent(question: str) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
    return graph.invoke({"question": question})


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    result = _run_agent(req.question)

    answer = result.get("answer", "")
    sources = [SourceItem(**s) for s in result.get("sources", [])]
    intent = result.get("intent", "general")

    SESSIONS.setdefault(session_id, []).append({"role": "user", "content": req.question})
    SESSIONS[session_id].append({"role": "assistant", "content": answer, "sources": [s.model_dump() for s in sources]})

    return ChatResponse(
        session_id=session_id,
        intent=intent,
        answer=answer,
        sources=sources,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE: emite la respuesta del agente token por token.

    Primero manda un evento 'sources' con las fuentes recuperadas, despues
    eventos 'token' con el texto. Al final un evento 'done'.
    """
    session_id = req.session_id or str(uuid.uuid4())

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    result = graph.invoke({"question": req.question})
    sources = result.get("sources", [])
    intent = result.get("intent", "general")
    answer = result.get("answer", "")

    # Guardamos en SESSIONS ANTES de empezar a streamear. Asi, si el cliente
    # corta la conexion, la conversacion ya queda persistida para el PDF.
    SESSIONS.setdefault(session_id, []).append({"role": "user", "content": req.question})
    SESSIONS[session_id].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

    def generate():
        yield f"event: meta\ndata: {json.dumps({'session_id': session_id, 'intent': intent})}\n\n"
        yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"

        chunk_size = 24
        for i in range(0, len(answer), chunk_size):
            yield f"event: token\ndata: {json.dumps({'text': answer[i:i + chunk_size]})}\n\n"

        yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --------------------------------------------------------------------
# Comparador RAG: corre las 6 strategies en paralelo
# --------------------------------------------------------------------

@app.post("/compare")
async def compare(req: CompareRequest) -> dict:
    """Corre las 6 strategies de retrieval en paralelo y devuelve todo lado a lado.

    Body: {question, history?, k?}
    Response: {
        question, strategies: {
            baseline: {answer, sources, items, metrics},
            hybrid:   {...},
            rerank:   {...},
            query_rewrite: {...},
            multi_query:   {...},
            hyde:     {...},
        }
    }
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")
    result = await run_compare(req.question, req.history, req.k or 6)
    return result.to_dict()


# --------------------------------------------------------------------
# Export PDF
# --------------------------------------------------------------------

@app.post("/export-pdf")
def export_pdf(req: ExportPdfRequest) -> StreamingResponse:
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail=f"sesion {req.session_id} no encontrada")
    messages = SESSIONS[req.session_id]
    pdf_bytes = render_conversation_pdf(messages, edition=EDITION)
    filename = f"agroposta_{req.session_id[:8]}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --------------------------------------------------------------------
# Dev: inspeccionar / borrar sesiones
# --------------------------------------------------------------------

@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="sesion no encontrada")
    return {"session_id": session_id, "messages": SESSIONS[session_id]}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return {"deleted": session_id}
