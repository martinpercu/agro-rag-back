"""Cliente HTTP para el servicio de cross-encoder rerank (Jina-compatible).

El servicio corre en la Mac mini (mlx-lm + FastAPI, puerto 8001) y expone
POST /v1/rerank con body:
    {"model": <opcional>, "query": str, "documents": [str, ...], "top_n": int}
Respuesta:
    {"model": str, "results": [{"index": int, "relevance_score": float}, ...]}

Configurable via env AGROPOSTA_RERANK_URL (default: Mac mini).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

DEFAULT_RERANK_URL = "http://192.168.12.215:8001/v1/rerank"
TIMEOUT_S = 60


@dataclass
class RerankResult:
    indices: list[int]  # indices dentro de documents, ordenados por relevancia
    scores: list[float]  # relevance_score por cada indice (mismo orden)
    model: str
    service_ms: float


class RerankClient:
    def __init__(self, base_url: str | None = None, timeout: float = TIMEOUT_S):
        self.base_url = (base_url or os.getenv("AGROPOSTA_RERANK_URL") or DEFAULT_RERANK_URL).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> RerankResult:
        """Rerankea documents contra query. Levanta RuntimeError si el servicio falla."""
        payload: dict = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        t0 = time.time()
        resp = self._client.post(self.base_url, json=payload)
        elapsed_ms = (time.time() - t0) * 1000
        if resp.status_code != 200:
            raise RuntimeError(
                f"rerank_service_http_{resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        results = data.get("results") or []
        indices = [r["index"] for r in results if isinstance(r.get("index"), int)]
        scores = [float(r["relevance_score"]) for r in results]
        return RerankResult(
            indices=indices,
            scores=scores,
            model=str(data.get("model", "")),
            service_ms=elapsed_ms,
        )

    def close(self) -> None:
        self._client.close()