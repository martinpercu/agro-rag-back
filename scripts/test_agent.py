"""Test end-to-end del agente en CLI."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.graph import graph

QUERIES = [
    "Cuanto me sale sembrar soja de primera en zona norte de Buenos Aires?",
    "Que se proyecta para trigo en la campana 2026/27?",
    "Cual es el margen bruto del maiz tardio?",
    "Que relacion insumo producto tiene el glifosato?",
    "Cuanto cuesta un kilo de novillo en feedlot?",
]


def main() -> int:
    for i, q in enumerate(QUERIES, 1):
        print("=" * 80)
        print(f"[{i}] PREGUNTA: {q}")
        print("-" * 80)
        result = graph.invoke({"question": q})
        print(f"Intent detectado: {result.get('intent')}")
        print(f"Chunks recuperados: {len(result.get('retrieved', []))}")
        print()
        print("RESPUESTA:")
        print(result.get("answer", "(sin respuesta)"))
        print()
        print("FUENTES:")
        for s in result.get("sources", []):
            print(
                f"  - pag. {s['pagina']} | seccion={s['seccion']} | "
                f"tipo={s['tipo']} | cultivo={s.get('cultivo') or '-'} | "
                f"campana={s.get('campana') or '-'} | score={s['score']}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
