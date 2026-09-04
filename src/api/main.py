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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import instrumentation  # noqa: F401  side-effect: OTel → Langfuse local si LANGFUSE_HOST (now no-op, see observability.py)

from observability import get_langfuse as _get_langfuse  # local-only 3003, no OTel

# Keep legacy var for backwards compat (plan_parse still checks it)
try:
    _langfuse_client = _get_langfuse()
except Exception:
    _langfuse_client = None

from agent.nodes.classifier import is_off_topic
from agent.strategies.runner import (
    get_all_strategies,
    get_strategies_by_names,
    run_compare,
    run_compare_stream,
)
from ingestion.vector_store import get_vector_store, vector_store_name

load_dotenv()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("AGROPOSTA_ALLOWED_ORIGINS", "http://localhost:3002").split(",") if o.strip()]
EDITION = os.getenv("AGROPOSTA_EDITION", "2026_05")

app = FastAPI(title="Agroposta API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_db_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return ""
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


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


class InvestigationCreate(BaseModel):
    query: str | None = None
    edition_id: str | None = None
    divisions: list[dict] | None = None
    location: dict | None = None
    price_variants: list[float] | None = None
    client_hint: str | None = None
    metadata: dict | None = None


class PlanCreate(BaseModel):
    investigation_id: str | None = None
    edition_id: str | None = None
    total_hectares: str | None = None
    season: str | None = None
    divisions: list[dict] | None = None
    location: dict | None = None
    price_variants: list[float] | None = None
    client_hint: str | None = None
    metadata: dict | None = None


class PlanParseRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] | None = None


class PlanParseResponse(BaseModel):
    plan_intent: bool
    divisions: list[dict]
    location: dict


# --------------------------------------------------------------------
# Endpoints basicos
# --------------------------------------------------------------------

@app.get("/")
def health() -> dict:
    vs = get_vector_store()
    return {"status": "ok", "service": "agroposta", "edition": EDITION, "vector_store": vector_store_name(), "vector_health": vs.health()}


@app.get("/stats")
def stats() -> dict:
    vs = get_vector_store()
    base = vs.collection_stats()
    base["vector_store"] = vector_store_name()
    return base


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

        return StreamingResponse(
            _reject(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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

    # Lab trace: lab:compare_stream (research, separado de user:chat)
    # Usa session_id="lab" fijo, tags=["lab"], no depende del usuario (lab es unico)
    lf = _get_langfuse()
    if lf is not None:
        # Wrap streaming in lab trace — context stays alive during async iteration
        async def _format_events():
            try:
                from observability import start_as_current_observation, update_current_observation, flush  # type: ignore

                with start_as_current_observation(
                    name="lab:compare_stream",
                    as_type="span",
                    input={
                        "question": req.question,
                        "history": req.history,
                        "enabled": [s.name for s in strategies],
                        "k": req.k or 6,
                    },
                    session_id="lab",
                    user_id="lab",
                    tags=["lab"],
                    metadata={"enabled": [s.name for s in strategies], "lang": req.lang},
                ) as _lab_span:
                    done_count = 0
                    error_count = 0
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
                            done_count += 1
                            payload = {"strategy": name, **data}
                            yield f"event: strategy_done\ndata: {json.dumps(payload)}\n\n"
                        elif typ == "error":
                            error_count += 1
                            yield f"event: strategy_error\ndata: {json.dumps({'strategy': name, 'error': data})}\n\n"
                    # Update lab trace output after stream finishes
                    try:
                        update_current_observation(
                            output={"done": done_count, "errors": error_count, "strategies": [s.name for s in strategies]}
                        )
                    except Exception:
                        pass
                    try:
                        flush()
                    except Exception:
                        pass
            except Exception:
                # Fallback without Langfuse if helper fails
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

        return StreamingResponse(
            _format_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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

    return StreamingResponse(
        _format_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------
# Auth + DB — Fase 0 (Supabase JWT + Railway Postgres)
# --------------------------------------------------------------------

def _ensure_user(db, payload: dict) -> dict:
    """Get or create user from Supabase payload. Returns DB user dict."""
    import uuid

    from db.models import User

    supa_id = payload.get("sub")
    email = payload.get("email") or payload.get("user_metadata", {}).get("email") if isinstance(payload.get("user_metadata"), dict) else payload.get("email")
    if not supa_id:
        raise HTTPException(status_code=401, detail="Token missing sub")
    try:
        uid = uuid.UUID(supa_id)
    except Exception:
        # Deterministic UUID for non-UUID subs (e.g. dev-user) — keeps same user across requests
        uid = uuid.uuid5(uuid.NAMESPACE_DNS, str(supa_id))

    # Try to find by supabase_user_id
    user = None
    try:
        user = db.query(User).filter(User.supabase_user_id == uid).first()
    except Exception:
        db.rollback()
        user = None
    if not user and supa_id:
        # Only try string search if supa_id is a valid UUID string to avoid
        # "invalid input syntax for type uuid" which aborts the transaction
        try:
            uuid.UUID(str(supa_id))
            user = db.query(User).filter(User.supabase_user_id == supa_id).first()  # type: ignore
        except Exception:
            db.rollback()
            pass
    if user:
        return {"id": str(user.id), "supabase_user_id": str(user.supabase_user_id), "email": user.email, "role": user.role, "allow_aggregated_use": user.allow_aggregated_use}

    # Create
    new_user = User(supabase_user_id=uid, email=email or f"{supa_id}@unknown", role="user", profile_hints={}, allow_aggregated_use=False)
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
    except Exception as e:
        db.rollback()
        # Race: try fetch again
        user = db.query(User).filter(User.supabase_user_id == uid).first()
        if user:
            return {"id": str(user.id), "supabase_user_id": str(user.supabase_user_id), "email": user.email, "role": user.role, "allow_aggregated_use": user.allow_aggregated_use}
        raise HTTPException(status_code=500, detail=f"User create failed: {e}")
    return {"id": str(new_user.id), "supabase_user_id": str(new_user.supabase_user_id), "email": new_user.email, "role": new_user.role, "allow_aggregated_use": new_user.allow_aggregated_use}


@app.get("/me")
async def me(request: Request):
    """Current user profile (requires Bearer). Returns user + stats."""
    from api.auth import get_current_user

    payload = await get_current_user(request)
    # Lazy DB import to avoid circular
    from sqlalchemy.orm import Session as SASession

    # get_db manually
    from db.session import get_db as _get_db
    from db.models import Investigation, Plan, Session as DBSession

    gen = _get_db()
    db_sess = next(gen)
    try:
        user = _ensure_user(db_sess, payload)
        # Counts
        inv_count = db_sess.query(Investigation).filter(Investigation.user_id == user["id"]).count()
        plan_count = db_sess.query(Plan).filter(Plan.user_id == user["id"]).count()
        sess_count = db_sess.query(DBSession).filter(DBSession.user_id == user["id"]).count()
        return {"user": user, "payload": {"sub": payload.get("sub"), "email": payload.get("email")}, "counts": {"investigations": inv_count, "plans": plan_count, "sessions": sess_count}}
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@app.get("/investigations")
async def list_investigations(request: Request):
    from api.auth import get_current_user
    from db.session import get_db as _get_db
    from db.models import Investigation

    payload = await get_current_user(request)
    gen = _get_db()
    db = next(gen)
    try:
        user = _ensure_user(db, payload)
        rows = db.query(Investigation).filter(Investigation.user_id == user["id"]).order_by(Investigation.created_at.desc()).limit(50).all()
        return {"investigations": [{"id": str(r.id), "query": r.query, "edition_id": r.edition_id, "divisions": r.divisions, "location": r.location, "price_variants": r.price_variants, "client_hint": r.client_hint, "metadata": r.meta, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]}
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@app.post("/investigations")
async def create_investigation(body: InvestigationCreate, request: Request):
    from api.auth import get_current_user
    from db.session import get_db as _get_db
    from db.models import Investigation

    payload = await get_current_user(request)
    gen = _get_db()
    db = next(gen)
    try:
        user = _ensure_user(db, payload)
        inv = Investigation(
            user_id=user["id"],
            edition_id=body.edition_id or EDITION,
            query=body.query,
            divisions=body.divisions or [],
            location=body.location or {},
            price_variants=body.price_variants or [],
            client_hint=body.client_hint,
            meta=body.metadata or {},
        )
        db.add(inv)
        db.commit()
        db.refresh(inv)
        return {"id": str(inv.id), "created_at": inv.created_at.isoformat() if inv.created_at else None}
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@app.get("/plans")
async def list_plans(request: Request):
    from api.auth import get_current_user
    from db.session import get_db as _get_db
    from db.models import Plan

    payload = await get_current_user(request)
    gen = _get_db()
    db = next(gen)
    try:
        user = _ensure_user(db, payload)
        rows = db.query(Plan).filter(Plan.user_id == user["id"]).order_by(Plan.created_at.desc()).limit(50).all()
        return {"plans": [{"id": str(r.id), "investigation_id": str(r.investigation_id) if r.investigation_id else None, "edition_id": r.edition_id, "total_hectares": r.total_hectares, "season": r.season, "divisions": r.divisions, "location": r.location, "price_variants": r.price_variants, "client_hint": r.client_hint, "metadata": r.meta, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]}
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


@app.post("/plans")
async def create_plan(body: PlanCreate, request: Request):
    from api.auth import get_current_user
    from db.session import get_db as _get_db
    from db.models import Plan

    payload = await get_current_user(request)
    gen = _get_db()
    db = next(gen)
    try:
        user = _ensure_user(db, payload)
        plan = Plan(
            user_id=user["id"],
            investigation_id=body.investigation_id,
            edition_id=body.edition_id or EDITION,
            total_hectares=body.total_hectares,
            season=body.season,
            divisions=body.divisions or [],
            location=body.location or {},
            price_variants=body.price_variants or [],
            client_hint=body.client_hint,
            meta=body.metadata or {},
        )
        db.add(plan)
        db.commit()
        db.refresh(plan)
        return {"id": str(plan.id), "created_at": plan.created_at.isoformat() if plan.created_at else None}
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


def _extract_session_id(request: Request | None = None) -> str:
    """Sin Auth: extrae session_id=user_id (supabase_user_id) del JWT si hay, si no anon.

    - Si Authorization: Bearer <jwt> presente, decodifica sin verificar y toma `sub`.
    - Si no, retorna "anon" (dev/local). Permite traza visible sin login.
    """
    if request is not None:
        try:
            auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
                if token:
                    # Try decode without verify (PyJWT)
                    try:
                        import jwt as _jwt

                        payload = _jwt.decode(token, options={"verify_signature": False})
                        sub = payload.get("sub")
                        if sub:
                            return str(sub)
                    except Exception:
                        pass
                    # Fallback: raw token as session_id (truncated)
                    return token[:64]
            # Also check x-user-id header for testing
            x_uid = request.headers.get("x-user-id") or request.headers.get("X-User-Id")
            if x_uid:
                return str(x_uid).strip()[:64]
        except Exception:
            pass
    return "anon"


@app.post("/plan/parse", response_model=PlanParseResponse)
def plan_parse(body: PlanParseRequest, request: Request) -> dict:
    """Parse ligero Fase 2 baby: detecta plan_intent + divisions sin LLM ni DB.

    Útil para el front antes de guardar investigada, y para tests.
    Traza manual a Langfuse si está configurado (local 3003) con session_id=user_id.
    """
    from agent.nodes.field_collector import extract_divisions
    from agent.nodes.plan_intent import is_plan_intent

    session_id = _extract_session_id(request)
    lf = _get_langfuse()
    if lf is not None:
        try:
            from langfuse import propagate_attributes  # type: ignore

            with propagate_attributes(session_id=session_id, user_id=session_id):
                with lf.start_as_current_observation(
                    name="plan_parse",
                    as_type="span",
                    input={"question": body.question, "history": body.history},
                    metadata={"session_id": session_id},
                ) as _span:
                    intent = is_plan_intent(body.question, body.history)
                    divisions = extract_divisions(body.question, body.history)
                    try:
                        lf.update_current_span(output={"plan_intent": intent, "divisions": divisions, "location": {}})
                    except Exception:
                        pass
                    # Also demonstrate full graph tracing: if called via graph, child spans will appear
                    # We keep lightweight (no LLM) but still use observability helper pattern.
                    # Flush so trace visible inmediatamente en Langfuse UI 3003
                    try:
                        lf.flush()
                    except Exception:
                        pass
                    return {"plan_intent": intent, "divisions": divisions, "location": {}}
        except Exception:
            pass
    intent = is_plan_intent(body.question, body.history)
    divisions = extract_divisions(body.question, body.history)
    # location vacío por ahora (DI-5 abierto)
    return {"plan_intent": intent, "divisions": divisions, "location": {}}


class GraphRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    history: list[dict] | None = None


@app.post("/chat")
async def chat(req: GraphRequest, request: Request) -> dict:
    """Chat producto: classifier→plan_intent→field_collector→retriever→answerer.

    Traza Langfuse 5 nodos con session_id=user_id (supabase_user_id o anon) — separada de lab.
    name="user:chat" tags no lab, session distingue usuarios autenticados en Langfuse Users/Sessions.
    Sin OTel, solo SDK manual.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    session_id = _extract_session_id(request)
    lf = _get_langfuse()

    # Fast path: with Langfuse trace user:chat (prod, por usuario)
    if lf is not None:
        try:
            from langfuse import propagate_attributes  # type: ignore

            with propagate_attributes(session_id=session_id, user_id=session_id):
                with lf.start_as_current_observation(
                    name="user:chat",
                    as_type="span",
                    input={"question": req.question, "history": req.history},
                    metadata={"session_id": session_id},
                ) as _root:
                    from agent.graph import graph

                    # Graph sync; run in thread to not block event loop
                    import asyncio

                    initial_state = {"question": req.question, "history": req.history or []}
                    # Prefer async ainvoke if available
                    try:
                        result = await graph.ainvoke(initial_state)
                    except Exception:
                        # Fallback sync
                        result = await asyncio.to_thread(graph.invoke, initial_state)

                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    try:
                        lf.update_current_span(
                            output={
                                "answer": answer,
                                "sources": sources,
                                "intent": result.get("intent"),
                                "plan_intent": result.get("plan_intent"),
                                "divisions": result.get("divisions"),
                            }
                        )
                    except Exception:
                        pass
                    try:
                        lf.flush()
                    except Exception:
                        pass
                    return {
                        "answer": answer,
                        "sources": sources,
                        "intent": result.get("intent"),
                        "plan_intent": result.get("plan_intent"),
                        "divisions": result.get("divisions"),
                        "location": result.get("location"),
                    }
        except Exception as e:
            # If Langfuse tracing fails, fallback to plain graph
            pass

    # No tracing or fallback
    from agent.graph import graph
    import asyncio

    initial_state = {"question": req.question, "history": req.history or []}
    try:
        result = await graph.ainvoke(initial_state)
    except Exception:
        result = await asyncio.to_thread(graph.invoke, initial_state)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "intent": result.get("intent"),
        "plan_intent": result.get("plan_intent"),
        "divisions": result.get("divisions"),
        "location": result.get("location"),
    }


@app.get("/editions")
def list_editions() -> dict:
    """List available Margenes editions (from Pinecone health + DB fallback)."""
    # For now single edition, but structure ready for multi-edition
    vs = get_vector_store()
    health = vs.health()
    return {"editions": [{"id": EDITION, "vector_health": health}], "current": EDITION}



