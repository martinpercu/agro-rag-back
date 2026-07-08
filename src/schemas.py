"""Schemas compartidos entre los modulos de ingestion, agent y api."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EditionId = str  # ej: "2026_05"
ChunkId = str  # uuid


class PageContent(BaseModel):
    """Lo que se extrae de una pagina del PDF."""

    page_number: int
    text: str
    tables: list[list[list[str]]] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    """Metadata que se persiste con cada chunk en el vector store."""

    edicion: EditionId
    seccion: str  # costos_margenes | analisis_mercado | proyecciones | tecnologia | ganaderia | insumos | general
    cultivo: str | None = None  # trigo | soja | maiz | girasol | sorgo | cebada | None
    zona: str | None = None  # libre; ej "norte_ba" | "nucleo" | "entre_rios"
    campana: str | None = None  # 2025_26 | 2026_27 | None
    tipo: Literal["tabla", "narrativa", "comentario_tecnico", "precio", "general"]
    pagina: int


class Chunk(BaseModel):
    """Un chunk listo para embedear."""

    id: ChunkId
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """Un chunk devuelto por la busqueda, con score."""

    chunk: Chunk
    score: float
