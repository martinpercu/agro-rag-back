"""Fixtures compartidos para los tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Permite imports absolutos tipo `from agent.graph import graph`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Carga .env desde la raiz del proyecto para que OPENAI_API_KEY este
# disponible en los tests de integracion.
load_dotenv(ROOT / ".env")


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="corre tests que pegan a OpenAI / ChromaDB real",
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def data_raw(project_root: Path) -> Path:
    return project_root / "data" / "raw"


@pytest.fixture(scope="session")
def pdf_path(data_raw: Path) -> Path:
    path = data_raw / "margenes_2026_05.pdf"
    if not path.exists():
        pytest.skip(f"PDF fuente no encontrado: {path}. Correr scripts/ingest_magazine.py primero.")
    return path


@pytest.fixture(scope="session")
def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture(scope="session")
def sample_pages():
    """Paginas sinteticas para tests del chunker sin tocar el PDF real."""
    from schemas import PageContent

    return [
        PageContent(
            page_number=1,
            text="Márgenes Agropecuarios, mayo 2026\nTRIGO: COSTOS y MARGENES\nZONA OESTE DE BUENOS AIRES\nRENDIMIENTOS QQ/ha 40\nPRECIO A COSECHA 2027 US$/tn 232",
            tables=[],
        ),
        PageContent(
            page_number=2,
            text="SOJA DE 1a.: COSTOS y MARGENES\nZONA NUCLEO\nRENDIMIENTOS QQ/ha 45\nPRECIO A COSECHA 2027 US$/tn 331",
            tables=[[["Cultivo", "Rinde", "Costo"], ["Soja 1a", "45", "538"]]],
        ),
        PageContent(
            page_number=3,
            text="CONTROL DE MALEZAS RESISTENTES\nMAIZ NORTE BS AS / SUR STA FE\nGLIFOSATO 54% US$/kg 5,0",
            tables=[],
        ),
        PageContent(
            page_number=4,
            text="TAMBO: COSTOS y MARGENES\nCRIA + RECRIA + FEEDLOT\nNOVILLO US$/kg 3,58",
            tables=[],
        ),
    ]


@pytest.fixture(scope="session")
def sample_chunks(sample_pages):
    """Chunks generados desde sample_pages para tests del retriever/indexer."""
    from ingestion.chunker import chunk_pages

    return chunk_pages(sample_pages, edition="2026_05")
