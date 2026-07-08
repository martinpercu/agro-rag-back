"""Chunker: convierte paginas extraidas en chunks con metadata para el vector store."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.extractor import PageContent, table_to_text
from schemas import Chunk, ChunkMetadata


# ------------------------------------------------------------------
# Deteccion de seccion, cultivo y campana desde el header de la pagina
# ------------------------------------------------------------------

# Orden importa: lo mas especifico primero.
SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "costos_margenes",
        re.compile(
            r"\b(MAIZ|SOJA|TRIGO|GIRASOL|SORGO|CEBADA|MA\xc3\x8dZ)\b"
            r"[^|\n]*\b(COSTOS\s+Y\s+M[ÁA]RGENES|COSTOS\s+Y\s+MARG\.?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "proyecciones",
        re.compile(
            r"\b(PRODUCCI[ÓO]N\s+Y\s+EXPORTACIONES|PROYECCIONES)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "analisis_mercado",
        re.compile(r"\b(AN[ÁA]LISIS\s+DEL\s+MERCADO\s+DE\s+GRANOS)\b", re.IGNORECASE),
    ),
    (
        "fas_teorico",
        re.compile(r"\b(FAS\s+TE[ÓO]RICO)\b", re.IGNORECASE),
    ),
    (
        "mercado_precios",
        re.compile(
            r"\b("
            r"ESTADO\s+DEL\s+MERCADO\s+PRESENTE\s+Y\s+FUTURO|"
            r"MERCADOS\s+INTERNACIONALES\s+A\s+T[ÉE]RMINO|"
            r"EVOLUCI[ÓO]N\s+DE\s+LOS\s+PRECIOS\s+EN\s+D[ÓO]LARES|"
            r"PRECIOS\s+HIST[ÓO]RICOS\s+Y\s+ACTUALES|"
            r"PRECIOS\s+DE\s+PRODUCTOS\s+E\s+INSUMOS\s+EN\s+D[ÓO]LARES|"
            r"RELACIONES\s+INSUMO\s*/\s*PRODUCTO|"
            r"PRECIOS,\s*COSTOS\s+Y\s+RETENCIONES|"
            r"TARIFAS\s+PARA\s+EL\s+TRANSPORTE\s+DE\s+GRANOS"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tecnologia",
        re.compile(
            r"\b("
            r"TECNOLOG[ÍI]A\s+DE\s+PUNTA|"
            r"CONTROL\s+DE\s+MALEZAS\s+RESISTENTES|"
            r"BASES\s+DE\s+PRESUPUESTACI[ÓO]N"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ganaderia_costos",
        re.compile(
            r"\b("
            r"INVERNADA:\s*COSTOS\s+Y\s+M[ÁA]RGENES|"
            r"CRIA:\s*COSTOS\s+Y\s+M[ÁA]RGENES|"
            r"CRIA:\s*CAPITALIZACI[ÓO]N\s+VS\.?\s+ARRENDAMIENTO|"
            r"TAMBO:\s*COSTOS\s+Y\s+M[ÁA]RGENES|"
            r"TAMBO:\s*EL\s+COSTO\s+POR\s+LITRO|"
            r"CICLO\s+COMPLETO|"
            r"LOS\s+N[ÚU]MEROS\s+DEL\s+FEEDLOT\s+CASERO|"
            r"COSTO\s+DE\s+VAQUILLONAS\s+DESDE\s+EL\s+NACIMIENTO\s+AL\s+PARTO|"
            r"COSTO\s+DE\s+SILO\s+DE\s+PASTURA|"
            r"COSTO\s+DE\s+SILAJE\s+DE\s+MA[ÍI]Z|"
            r"COSTO\s+DE\s+PASTURAS\s+Y\s+VERDEOS|"
            r"COSTO\s+DE\s+ROLLOS|"
            r"COSTO\s+DE\s+EMBOLSADO"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "siembras",
        re.compile(
            r"\b("
            r"SIEMBRAS\s+A\s+PORCENTAJE|"
            r"CAMPO:\s*SIEMBRAS\s+A\s+PORCENTAJE|"
            r"ARRENDAMIENTOS\s+AGR[ÍI]COLAS|"
            r"PLANTEOS\s+COMPARATIVOS"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "insumos_maquinaria",
        re.compile(
            r"\b("
            r"SEMILLAS\s+Y\s+AGROQU[ÍI]MICOS|"
            r"PROD\.?\s*VETERINARIOS|"
            r"TRACTORES\s+Y\s+COSECHADORAS|"
            r"MAQUINARIA\s+AGR[ÍI]COLA|"
            r"ART[ÍI]CULOS\s+RURALES"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "costos_operativos",
        re.compile(
            r"\b("
            r"COSTOS\s+Y\s+TARIFAS\s+DE\s+TRILLA|"
            r"COSTO\s+DE\s+SIEMBRA\s+Y\s+PULVERIZACI[ÓO]N|"
            r"COSTO\s+DE\s+COSECHA|"
            r"EL\s+COSTO\s+DE\s+COSECHA|"
            r"DETALLE\s+DE\s+GASTOS\s+DE\s+ESTRUCTURA|"
            r"COSTOS\s+Y\s+MARGEN\b(?!\s*ES)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "analisis_especial",
        re.compile(
            r"\b("
            r"LO\s+QUE\s+NOS\s+DICE\s+LA\s+MANGA|"
            r"PARTICULARIDADES\s+DE\s+UNA\s+ECONOM[ÍI]A\s+BIMONETARIA|"
            r"CUOTA\s+DE\s+REALISMO|"
            r"DIN[ÁA]MICA\s+DEL\s+FAS\s+TE[ÓO]RICO"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]


CULTIVO_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("trigo", re.compile(r"\bTRIGO\b", re.IGNORECASE)),
    ("soja", re.compile(r"\bSOJA\b", re.IGNORECASE)),
    ("maiz", re.compile(r"\bMA[ÍI]Z\b", re.IGNORECASE)),
    ("girasol", re.compile(r"\bGIRASOL\b", re.IGNORECASE)),
    ("sorgo", re.compile(r"\bSORGO\b", re.IGNORECASE)),
    ("cebada", re.compile(r"\bCEBADA\b", re.IGNORECASE)),
]


CAMPANA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("2026_27", re.compile(r"\b2026\s*/\s*27\b")),
    ("2025_26", re.compile(r"\b2025\s*/\s*26\b")),
    ("2024_25", re.compile(r"\b2024\s*/\s*25\b")),
]


# ------------------------------------------------------------------
# Helpers de deteccion
# ------------------------------------------------------------------

def _detect_section(text: str) -> str:
    """Detecta la seccion de la pagina a partir de las primeras lineas."""
    head = "\n".join(text.splitlines()[:30])
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(head):
            return name
    return "general"


def _detect_cultivo(text: str, section: str) -> str | None:
    """Si la seccion es costos/proyecciones de un cultivo especifico, lo devuelve."""
    if section not in {"costos_margenes", "proyecciones", "mercado_precios"}:
        return None
    head = "\n".join(text.splitlines()[:20])
    # Prioridad: si la pagina menciona varios (ej. complejo soja + girasol), agarramos el primero
    for name, pattern in CULTIVO_PATTERNS:
        if pattern.search(head):
            return name
    return None


def _detect_campana(text: str) -> str | None:
    for name, pattern in CAMPANA_PATTERNS:
        if pattern.search(text):
            return name
    return None


# ------------------------------------------------------------------
# Chunker principal
# ------------------------------------------------------------------

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(
    pages: list[PageContent],
    edition: str,
) -> list[Chunk]:
    """Convierte paginas en chunks con metadata.

    Reglas:
    - Si la pagina tiene tablas, se hace 1 chunk atomico: texto de la
      pagina + todas las tablas. Esto preserva la coherencia entre el
      titulo de seccion, comentarios y planilla.
    - Si la pagina es solo narrativa, se parte con recursive splitter.
    - El tipo del chunk es "tabla" si la pagina tiene tablas, sino
      "narrativa" o "comentario_tecnico" segun la seccion.
    """
    chunks: list[Chunk] = []
    for page in pages:
        section = _detect_section(page.text)
        cultivo = _detect_cultivo(page.text, section)
        campana = _detect_campana(page.text)

        if page.tables:
            chunks.extend(
                _chunks_from_table_page(page, edition, section, cultivo, campana)
            )
        else:
            chunks.extend(
                _chunks_from_narrative_page(page, edition, section, cultivo, campana)
            )

    # Deduplicar chunks con texto identico (pdfplumber suele fragmentar una
    # misma planilla en varios sub-pedazos que se vuelven chunks duplicados).
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for c in chunks:
        if c.text in seen:
            continue
        seen.add(c.text)
        deduped.append(c)
    return deduped


def _chunks_from_table_page(
    page: PageContent,
    edition: str,
    section: str,
    cultivo: str | None,
    campana: str | None,
) -> list[Chunk]:
    """Una pagina con tablas es UN solo chunk atomico.

    pdfplumber suele fragmentar una misma planilla visual en N sub-tablas.
    Si las partiéramos por tabla, terminariamos con N chunks casi-identicos
    que ensucian el retrieval. Mejor consolidarlas: la pagina entera es
    el chunk. gpt-4o-mini aguanta 12k chars de contexto.
    """
    tipo: Literal["tabla", "narrativa", "comentario_tecnico", "precio", "general"] = "tabla"
    parts: list[str] = [f"=== {edition.upper()} | Pág. {page.page_number} ==="]
    if page.text.strip():
        parts.append(page.text.strip())
    for i, table in enumerate(page.tables, start=1):
        rendered = table_to_text(table)
        parts.append(f"\n--- Tabla {i} ---\n{rendered}")

    full_text = "\n\n".join(parts).strip()
    return [_make_chunk(full_text, edition, section, cultivo, campana, tipo, page.page_number)]


def _chunks_from_narrative_page(
    page: PageContent,
    edition: str,
    section: str,
    cultivo: str | None,
    campana: str | None,
) -> list[Chunk]:
    tipo: Literal["tabla", "narrativa", "comentario_tecnico", "precio", "general"]
    if section in {"tecnologia", "ganaderia_costos", "analisis_especial", "proyecciones"}:
        tipo = "comentario_tecnico"
    elif section == "mercado_precios":
        tipo = "precio"
    else:
        tipo = "narrativa"

    text = page.text.strip()
    if not text:
        return []

    header = f"=== {edition.upper()} | Pág. {page.page_number} ===\n"
    pieces = _SPLITTER.split_text(text)
    return [
        _make_chunk(header + piece, edition, section, cultivo, campana, tipo, page.page_number)
        for piece in pieces
        if piece.strip()
    ]


def _make_chunk(
    text: str,
    edition: str,
    section: str,
    cultivo: str | None,
    campana: str | None,
    tipo: Literal["tabla", "narrativa", "comentario_tecnico", "precio", "general"],
    page: int,
) -> Chunk:
    metadata = ChunkMetadata(
        edicion=edition,
        seccion=section,
        cultivo=cultivo,
        zona=None,
        campana=campana,
        tipo=tipo,
        pagina=page,
    )
    return Chunk(id=str(uuid.uuid4()), text=text, metadata=metadata)
