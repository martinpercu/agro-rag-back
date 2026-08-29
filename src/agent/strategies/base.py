"""Interface comun para todas las estrategias de retrieval."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceStep:
    """Un paso del pipeline de retrieval de una strategy, para el trace."""

    step: str  # ej: "classify", "embed_query", "chroma_search", "llm_rerank"
    detail: str = ""  # ej: "intent=ganaderia, filtro=[ganaderia_costos]"
    ms: float = 0.0  # duracion REAL del paso (ms)
    acc_ms: float = 0.0  # acumulado desde el inicio del retrieve (ms)
    at: str = ""  # ISO timestamp UTC (con ms) del momento en que termino el paso

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "detail": self.detail,
            "ms": round(self.ms, 1),
            "acc_ms": round(self.acc_ms, 1),
            "at": self.at,
        }


class TraceRecorder:
    """Registra pasos con duracion real y acumulado desde el inicio del retrieve.

    `step()` con `at_t` (time.monotonic del fin del paso) permite registrar
    pasos que terminaron en el pasado (ej: pasos internos de un helper que
    devuelve sus timings), y el acumulado queda consistente.
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._t_prev = self._t0
        self._wall0 = time.time()
        self.steps: list[TraceStep] = []

    def step(self, step: str, detail: str = "", at_t: float | None = None) -> None:
        t = time.monotonic() if at_t is None else at_t
        if t < self._t_prev:
            t = self._t_prev
        self.steps.append(
            TraceStep(
                step=step,
                detail=detail,
                ms=(t - self._t_prev) * 1000,
                acc_ms=(t - self._t0) * 1000,
                at=datetime.fromtimestamp(
                    self._wall0 + (t - self._t0), tz=timezone.utc
                ).isoformat(timespec="milliseconds"),
            )
        )
        self._t_prev = t

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]


@dataclass
class RetrievedItem:
    """Un chunk devuelto por la estrategia, con su score y metadata de origen."""

    chunk_id: str
    text: str
    seccion: str | None
    pagina: int | None
    cultivo: str | None
    campana: str | None
    tipo: str | None
    score: float  # similitud coseno o RRF score
    rank: int  # posicion en el ranking de la estrategia (0 = top)


@dataclass
class StrategyResult:
    """Lo que devuelve una estrategia despues de recuperar."""

    name: str
    items: list[RetrievedItem] = field(default_factory=list)
    intent: str = "general"
    retrieval_ms: float = 0.0
    llm_input_tokens: int = 0  # tokens de las llamadas LLM previas al retrieval
    llm_output_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)

    @property
    def num_sources(self) -> int:
        return len(self.items)

    @property
    def distinct_sources(self) -> int:
        """Cantidad de fuentes unicas por (seccion, pagina)."""
        return len({(i.seccion, i.pagina) for i in self.items})

    def trace_to_dict(self) -> list[dict]:
        return [t.to_dict() for t in self.trace]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "intent": self.intent,
            "retrieval_ms": round(self.retrieval_ms, 2),
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "num_sources": self.num_sources,
            "distinct_sources": self.distinct_sources,
            "trace": self.trace_to_dict(),
            "items": [
                {
                    "chunk_id": i.chunk_id,
                    "seccion": i.seccion,
                    "pagina": i.pagina,
                    "cultivo": i.cultivo,
                    "campana": i.campana,
                    "tipo": i.tipo,
                    "score": round(i.score, 4),
                    "rank": i.rank,
                    "text": i.text,
                }
                for i in self.items
            ],
            "extra": self.extra,
        }


class Strategy(ABC):
    """Interface para una estrategia de retrieval.

    Una estrategia recibe la pregunta cruda y (opcionalmente) el historial
    de la conversacion, y devuelve un StrategyResult con los chunks que
    encontro y metadata de como los encontro.

    El answerer se corre aparte (en el runner) usando los chunks que
    devolvio cada estrategia. Asi se aísla la diferencia: solo cambia
    el retrieval, el prompt y el LLM son los mismos para todas.
    """

    name: str = "base"

    @abstractmethod
    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = 6,
    ) -> StrategyResult:
        """Recupera hasta k chunks para la pregunta dada.

        history: lista opcional de {role, content} con los ultimos mensajes
        de la conversacion. Las estrategias que usan LLM antes del
        retrieval (rewriting, hyde, multi_query) lo consumen aca.
        """


def time_call(fn, *args, **kwargs):
    """Helper: corre fn() y devuelve (resultado, ms_transcurridos)."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    return result, (time.time() - t0) * 1000
