"""Bakeoff de modelos de embedding: golden questions x (baseline, hybrid).

Mide, por modelo de embedding, la calidad del retrieval AISLADA del LLM:
- baseline (semantica pura) y hybrid (BM25 + semantica) no llaman al LLM,
  asi que las diferencias vienen solo del embedding.
- Metricas: match de seccion esperada, match de cultivo esperado,
  latencia media del retrieval, fuentes devueltas.

Uso:
    # Correr contra LM Studio (mac mini): primer modelar el env como en .env.local
    uv run python scripts/embedding_bakeoff.py --models text-embedding-nomic-embed-text-v1.5

    # Correr varios modelos en un solo proceso (switchea AGROPOSTA_EMBEDDING_MODEL)
    uv run python scripts/embedding_bakeoff.py \
        --models text-embedding-nomic-embed-text-v1.5,text-embedding-3-small

Nota: para text-embedding-3-small (OpenAI) se necesita la API key real en
.env (no la dummy de LM Studio) y no debe estar set AGROPOSTA_LLM_BASE_URL.
Nota: cada proceso corre contra UN solo servidor de embeddings
(AGROPOSTA_EMBEDDINGS_BASE_URL, o el LLM base URL por fallback). Si los
modelos viven en servidores distintos (bge-m3 en :8001, nomic en LM
Studio), correrlos en invocaciones separadas con su env, no juntos.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.llm import DEFAULT_EMBEDDING_MODEL, collection_name
from agent.strategies.baseline import BaselineStrategy
from agent.strategies.hybrid import HybridStrategy
from agent.strategies.base import Strategy

GOLDEN_PATH = ROOT / "tests" / "golden_questions.json"
K = 6


def _load_golden() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _score_question(strategy: Strategy, q: dict) -> dict:
    t0 = time.time()
    result = strategy.retrieve(q["question"], k=K)
    ms = (time.time() - t0) * 1000
    items = result.items
    sections = {i.seccion for i in items}
    cultivos = {i.cultivo for i in items if i.cultivo}

    expected_sections = set(q.get("expected_sections_any") or [])
    expected_cultivos = set(c for c in (q.get("expected_cultivo_any") or []) if c)

    return {
        "id": q["id"],
        "intent": result.intent,
        "expected_intent": q.get("expected_intent"),
        "sections_ok": bool(expected_sections & sections) if expected_sections else None,
        "cultivo_ok": bool(expected_cultivos & cultivos) if expected_cultivos else None,
        "num_sources": len(items),
        "retrieval_ms": ms,
        "error": result.extra.get("error"),
    }


def _run_model(model: str, strategies: list[tuple[str, Strategy]], golden: list[dict]) -> dict:
    # El default de OpenAI requiere la API oficial, no el servidor local
    if model == DEFAULT_EMBEDDING_MODEL:
        os.environ.pop("AGROPOSTA_LLM_BASE_URL", None)
        os.environ.pop("AGROPOSTA_EMBEDDINGS_BASE_URL", None)
    os.environ["AGROPOSTA_EMBEDDING_MODEL"] = model

    col = collection_name(model)
    rows = []
    for sname, strategy in strategies:
        for q in golden:
            row = _score_question(strategy, q)
            row["strategy"] = sname
            rows.append(row)

    def _rate(rows_, key):
        vals = [r[key] for r in rows_ if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None

    def _avg(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    per_strategy = {}
    for sname, _ in strategies:
        sub = [r for r in rows if r["strategy"] == sname]
        per_strategy[sname] = {
            "sections_ok": _rate(sub, "sections_ok"),
            "cultivo_ok": _rate(sub, "cultivo_ok"),
            "avg_retrieval_ms": _avg(sub, "retrieval_ms"),
            "avg_sources": _avg(sub, "num_sources"),
            "errors": [r["id"] for r in sub if r.get("error")],
        }

    return {
        "model": model,
        "collection": col,
        "dims": None,  # se llena en el report con una query de prueba
        "per_strategy": per_strategy,
        "rows": rows,
    }


def _measure_dims(model: str) -> int | None:
    from agent.llm import get_embeddings

    try:
        return len(get_embeddings(model).embed_query("dims de prueba"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="text-embedding-nomic-embed-text-v1.5",
        help="Modelos separados por coma",
    )
    parser.add_argument(
        "--strategies",
        default="baseline,hybrid",
        help="Estrategias separadas por coma (default: baseline,hybrid)",
    )
    parser.add_argument("--json", action="store_true", help="Imprimir tambien JSON completo")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]

    available = {
        "baseline": ("baseline", BaselineStrategy()),
        "hybrid": ("hybrid", HybridStrategy()),
    }
    strategies = [available[s] for s in strategy_names if s in available]

    golden = _load_golden()
    print(f"Golden questions: {len(golden)} | strategies: {[s[0] for s in strategies]}")
    print()

    results = []
    for model in models:
        print(f"== {model} ==")
        r = _run_model(model, strategies, golden)
        r["dims"] = _measure_dims(model)
        results.append(r)
        print(f"   coleccion: {r['collection']} | dims: {r['dims']}")
        for sname, s in r["per_strategy"].items():
            print(
                f"   {sname:9s} seccion_ok={_pct(s['sections_ok']):6} "
                f"cultivo_ok={_pct(s['cultivo_ok']):6} "
                f"retrieval={s['avg_retrieval_ms']:7.1f}ms fuentes={s['avg_sources']:4.1f} "
                f"errors={s['errors']}"
            )
        print()

    print("=" * 60)
    print("RESUMEN (seccion_ok / cultivo_ok / retrieval_ms)")
    print(f"{'modelo':38s} {'baseline':>28s} {'hybrid':>28s}")
    for r in results:
        b = r["per_strategy"].get("baseline")
        h = r["per_strategy"].get("hybrid")
        b_txt = f"{_pct(b['sections_ok'])}/{_pct(b['cultivo_ok'])}/{b['avg_retrieval_ms']:.0f}ms" if b else "-"
        h_txt = f"{_pct(h['sections_ok'])}/{_pct(h['cultivo_ok'])}/{h['avg_retrieval_ms']:.0f}ms" if h else "-"
        print(f"{r['model']:38s} {b_txt:>28s} {h_txt:>28s}")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _pct(v: float | None) -> str:
    return f"{v * 100:.0f}%" if v is not None else "-"


if __name__ == "__main__":
    raise SystemExit(main())