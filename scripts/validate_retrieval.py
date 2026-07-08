"""Test rapido del retriever: corre 4 queries y muestra los top chunks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.indexer import collection_stats, search

QUERIES = [
    "costos de produccion de soja en zona nucleo",
    "que se proyecta para trigo campana 2026/27",
    "margen bruto del maiz tardio",
    "relacion insumo producto del glifosato",
]


def main() -> int:
    print("== Stats del vector store ==")
    stats = collection_stats()
    print(f"  total chunks: {stats['total']}")
    print()
    for i, q in enumerate(QUERIES, 1):
        print(f"[{i}] Query: {q}")
        results = search(q, k=3)
        for j, (chunk, score) in enumerate(results, 1):
            meta = chunk.metadata
            tipo = meta.tipo
            seccion = meta.seccion
            cultivo = meta.cultivo or "-"
            pagina = meta.pagina
            text = chunk.text.replace("\n", " ")[:220]
            print(f"    ({j}) score={score:.3f} | {seccion}/{tipo} | {cultivo} p{pagina}")
            print(f"        {text}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
