"""Genera un reporte markdown del comparador RAG.

Corre las golden questions contra las 6 strategies y produce una tabla
markdown con latencias, fuentes y tokens. Pensado para revision manual
NO para CI (las golden questions del pytest son el regression formal).

Uso:
    uv run python scripts/compare_report.py
    # -> imprime tabla en stdout
    # -> guarda el reporte completo en tmp/compare_report.md
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GOLDEN_PATH = ROOT / "tests" / "golden_compare_questions.json"
OUTPUT_PATH = ROOT / "tmp" / "compare_report.md"


def _format_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _format_ms(ms: float) -> str:
    if ms < 1000:
        return f"{int(ms)}ms"
    return f"{ms/1000:.1f}s"


def _strategy_row(name: str, sr) -> str:
    m = sr.metrics
    if m.get("error"):
        return f"| {name} | ERROR | - | - | - | - | {m['error'][:30]} |"
    retrieval = _format_ms(m.get("retrieval_ms", 0))
    answerer = _format_ms(m.get("answerer_ms", 0))
    total = _format_ms(m.get("total_ms", 0))
    sources = f"{m.get('num_sources', 0)}/{m.get('distinct_sources', 0)}"
    ans_tok = f"{_format_int(m.get('answerer_input_tokens', 0))}+{_format_int(m.get('answerer_output_tokens', 0))}"
    aux_tok = m.get("aux_llm_input_tokens", 0) + m.get("aux_llm_output_tokens", 0)
    aux_str = f"+{aux_tok} aux" if aux_tok > 0 else ""
    return f"| {name} | {total} | {retrieval} | {answerer} | {sources} | {ans_tok} {aux_str} | - |"


def _question_table(case: dict, result) -> str:
    lines = []
    lines.append(f"### `{case['id']}`: {case['question']}")
    lines.append("")
    lines.append(f"_Intent esperado: `{case['expected_intent']}` · {case['description']}_")
    lines.append("")
    lines.append("| Strategy | Total | Retrieval | Answerer | Sources | Tokens | Intent | Status |")
    lines.append("|----------|------:|----------:|---------:|--------:|-------:|--------|--------|")
    for name, sr in result.strategies.items():
        m = sr.metrics
        intent = m.get("intent", "?")
        err = m.get("error")
        status = f"❌ {err[:40]}" if err else "✅"
        lines.append(
            f"| {name} | {_format_ms(m.get('total_ms', 0))} | "
            f"{_format_ms(m.get('retrieval_ms', 0))} | "
            f"{_format_ms(m.get('answerer_ms', 0))} | "
            f"{m.get('num_sources', 0)}/{m.get('distinct_sources', 0)} | "
            f"{_format_int(m.get('answerer_input_tokens', 0))}+{_format_int(m.get('answerer_output_tokens', 0))} | "
            f"{intent} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_table(per_question_results: list[tuple[dict, object]]) -> str:
    """Tabla resumen: promedios de latencia y tokens por strategy."""
    strategy_names = list(per_question_results[0][1].strategies.keys())
    totals = {n: {"retrieval_ms": [], "answerer_ms": [], "total_ms": [],
                  "answerer_input": [], "answerer_output": [],
                  "aux_input": [], "aux_output": [], "sources": [],
                  "distinct": [], "errors": 0}
              for n in strategy_names}

    for case, result in per_question_results:
        for n, sr in result.strategies.items():
            m = sr.metrics
            if m.get("error"):
                totals[n]["errors"] += 1
                continue
            totals[n]["retrieval_ms"].append(m.get("retrieval_ms", 0))
            totals[n]["answerer_ms"].append(m.get("answerer_ms", 0))
            totals[n]["total_ms"].append(m.get("total_ms", 0))
            totals[n]["answerer_input"].append(m.get("answerer_input_tokens", 0))
            totals[n]["answerer_output"].append(m.get("answerer_output_tokens", 0))
            totals[n]["aux_input"].append(m.get("aux_llm_input_tokens", 0))
            totals[n]["aux_output"].append(m.get("aux_llm_output_tokens", 0))
            totals[n]["sources"].append(m.get("num_sources", 0))
            totals[n]["distinct"].append(m.get("distinct_sources", 0))

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0

    lines = []
    lines.append("## Resumen (promedios sobre las golden questions)")
    lines.append("")
    lines.append("| Strategy | Avg Total | Avg Retrieval | Avg Answerer | Avg Tok ans in | Avg Tok ans out | Avg Aux | Sources | Errors |")
    lines.append("|----------|----------:|--------------:|-------------:|----------------:|-----------------:|--------:|--------:|-------:|")
    for n in strategy_names:
        t = totals[n]
        n_total = len(t["total_ms"])
        if n_total == 0:
            lines.append(f"| {n} | - | - | - | - | - | - | - | {t['errors']} |")
            continue
        aux_total = (sum(t["aux_input"]) + sum(t["aux_output"])) / n_total
        lines.append(
            f"| {n} | "
            f"{_format_ms(_avg(t['total_ms']))} | "
            f"{_format_ms(_avg(t['retrieval_ms']))} | "
            f"{_format_ms(_avg(t['answerer_ms']))} | "
            f"{int(_avg(t['answerer_input']))} | "
            f"{int(_avg(t['answerer_output']))} | "
            f"{int(aux_total)} | "
            f"{_avg(t['sources']):.1f} | "
            f"{t['errors']} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    from agent.strategies.runner import run_compare

    golden = json.loads(GOLDEN_PATH.read_text())

    print(f"Corriendo {len(golden)} golden questions contra las 6 strategies...")
    print()

    per_question: list[tuple[dict, object]] = []
    t_start = time.time()

    for case in golden:
        print(f">> {case['id']} ({case['question'][:60]}...)")
        t0 = time.time()
        result = await run_compare(case["question"])
        elapsed = time.time() - t0
        print(f"   OK en {elapsed:.1f}s")
        per_question.append((case, result))

    total_elapsed = time.time() - t_start

    # Construir el reporte
    parts = []
    parts.append("# Reporte del comparador RAG")
    parts.append("")
    parts.append(f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M')}")
    parts.append(f"**Golden questions:** {len(golden)}")
    parts.append(f"**Tiempo total:** {_format_ms(total_elapsed * 1000)}")
    parts.append("")
    parts.append("---")
    parts.append("")

    for case, result in per_question:
        parts.append(_question_table(case, result))
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(_summary_table(per_question))

    report = "\n".join(parts)

    # Imprimir a stdout
    print()
    print(report)

    # Guardar en disco
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)
    print()
    print(f"Reporte guardado en: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
