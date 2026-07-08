#!/usr/bin/env python
"""Ingesta de una edicion de Margenes Agropecuarios al vector store.

Uso:
    uv run python scripts/ingest_magazine.py data/raw/margenes_2026_05.pdf
    uv run python scripts/ingest_magazine.py data/raw/margenes_2026_05.pdf --edition 2026_05
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# Permite imports absolutos tipo `from schemas import ...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion.chunker import chunk_pages
from ingestion.extractor import extract_pdf
from ingestion.indexer import collection_stats, get_chroma_path, index_chunks


def infer_edition(pdf_path: Path) -> str:
    """Infiere el ID de edicion desde el nombre del archivo.

    Espera patron margenes_YYYY_MM.pdf. Si no, devuelve "desconocida".
    """
    m = re.search(r"(\d{4})[_-](\d{2})", pdf_path.stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return "desconocida"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path al PDF a ingestar")
    parser.add_argument(
        "--edition",
        type=str,
        default=None,
        help="ID de edicion (ej 2026_05). Si no se pasa, se infiere del nombre.",
    )
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(f"ERROR: PDF no encontrado: {pdf_path}", file=sys.stderr)
        return 1

    edition = args.edition or infer_edition(pdf_path)
    print(f"[1/3] Extrayendo texto y tablas de {pdf_path.name} (edicion: {edition})...")

    t0 = time.time()
    pages = extract_pdf(pdf_path)
    print(f"      {len(pages)} paginas extraidas en {time.time() - t0:.1f}s")

    print(f"[2/3] Chunking con metadata...")
    t0 = time.time()
    chunks = chunk_pages(pages, edition=edition)
    n_tablas = sum(1 for c in chunks if c.metadata.tipo == "tabla")
    n_narr = sum(1 for c in chunks if c.metadata.tipo == "narrativa")
    n_com = sum(1 for c in chunks if c.metadata.tipo == "comentario_tecnico")
    n_pre = sum(1 for c in chunks if c.metadata.tipo == "precio")
    print(
        f"      {len(chunks)} chunks generados en {time.time() - t0:.1f}s "
        f"({n_tablas} tablas, {n_narr} narrativos, {n_com} comentarios, {n_pre} precios)"
    )

    print(f"[3/3] Indexando en ChromaDB...")
    t0 = time.time()
    n_indexed = index_chunks(chunks, base=get_chroma_path())
    print(f"      {n_indexed} chunks nuevos indexados en {time.time() - t0:.1f}s")

    print("\n[stats] Estado del vector store:")
    stats = collection_stats(base=get_chroma_path())
    print(f"  total: {stats['total']}")
    print(f"  por seccion: {stats['by_section']}")
    print(f"  por tipo: {stats['by_tipo']}")
    print(f"  por cultivo: {stats['by_cultivo']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
