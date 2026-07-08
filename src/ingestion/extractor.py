"""Extraccion de texto y tablas desde un PDF de la revista Margenes Agropecuarios."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from schemas import PageContent


def extract_pdf(pdf_path: Path) -> list[PageContent]:
    """Devuelve una lista de paginas con texto y tablas separadas.

    El PDF de Margenes tiene planillas multi-columna muy densas. Por eso
    conservamos el texto lineal Y las tablas estructuradas por separado:
    el chunker decide como tratar cada pagina segun la seccion.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    pages: list[PageContent] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = _clean_text(text)

            tables_raw = page.extract_tables() or []
            tables = [
                _clean_table(t)
                for t in tables_raw
                if t
                and any(any(c and c.strip() for c in row) for row in t)
                and sum(1 for row in t if any(c and c.strip() for c in row)) >= 3
            ]

            pages.append(PageContent(
                page_number=page.page_number,
                text=text,
                tables=tables,
            ))
    return pages


def _clean_text(text: str) -> str:
    """Limpia artefactos tipicos del OCR/layout de Margenes."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    for ln in lines:
        if not ln.strip():
            cleaned.append("")
            continue
        cleaned.append(ln)
    out = "\n".join(cleaned)
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip()


def _clean_table(table: list[list[str | None]]) -> list[list[str]]:
    """Normaliza celdas None/Whitespace a string vacio."""
    return [[(cell or "").strip() for cell in row] for row in table]


def table_to_text(table: list[list[str]]) -> str:
    """Convierte una tabla extraida a texto con separadores para el LLM."""
    if not table:
        return ""
    widths = [max(len(cell) for cell in col) for col in zip(*table)]
    lines: list[str] = []
    for row in table:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append(" | ".join(padded).rstrip())
    return "\n".join(lines)
