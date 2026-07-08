"""Genera un PDF de una conversacion Agroposta."""
from __future__ import annotations

import io
from datetime import datetime

from fpdf import FPDF

AGROPOSTA_TITLE = "Agroposta - Informe de conversacion"
FOOTER_TEXT = "Generado por Agroposta - RAG sobre Margenes Agropecuarios"


def _safe(text: str) -> str:
    """fpdf2 latin-1 no soporta acentos ni enie. Los reemplazamos para que
    el PDF no rompa en runtime. Para MVP 1 alcanza; mas adelante podemos
    usar fpdf2 con fuentes unicode (dejavu) si se quiere.
    """
    if not text:
        return ""
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2026", "...")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
        .replace("\u00b7", "*")
        .replace("\u00b0", "o")
        .replace("\u20ac", "EUR")
    )


def render_conversation_pdf(
    messages: list[dict],
    edition: str = "2026_05",
) -> bytes:
    """Devuelve los bytes del PDF listo para descargar.

    messages: lista de {role: "user"|"assistant", content: str, sources?: list[dict]}
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe(AGROPOSTA_TITLE), ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe(f"Edicion consultada: Margenes Agropecuarios {edition.replace('_', '/')}"), ln=1)
    pdf.cell(0, 6, _safe(f"Fecha de exportacion: {datetime.now().strftime('%Y-%m-%d %H:%M')}"), ln=1)
    pdf.ln(4)

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        sources = msg.get("sources") or []

        if role == "user":
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(20, 80, 30)
            pdf.cell(0, 7, _safe("Productor:"), ln=1)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, _safe(content))
            pdf.ln(2)
        elif role == "assistant":
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(60, 30, 90)
            pdf.cell(0, 7, _safe("Agroposta:"), ln=1)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, _safe(content))
            pdf.ln(2)
            if sources:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(90, 90, 90)
                pdf.cell(0, 6, _safe("Fuentes citadas en la revista:"), ln=1)
                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Helvetica", "", 9)
                for s in sources:
                    parts = [
                        f"pag. {s.get('pagina', '?')}",
                        f"seccion {s.get('seccion', '?')}",
                    ]
                    if s.get("cultivo"):
                        parts.append(f"cultivo {s['cultivo']}")
                    if s.get("campana"):
                        parts.append(f"campana {s['campana'].replace('_', '/')}")
                    line = " - ".join(parts)
                    pdf.cell(0, 5, _safe(line), ln=1)
                pdf.ln(3)

    pdf.set_y(-12)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, _safe(FOOTER_TEXT), align="C")

    return bytes(pdf.output(dest="S"))
