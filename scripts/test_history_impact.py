"""Test del impacto del historial en query_rewrite.

Cuatro escenarios sobre DOS preguntas base:

Pregunta 1 (clara, "Cuanto cuesta un kilo de novillo en feedlot?"):
a) sin history
b) con history neutral (feedlot en zona nucleo, campo propio)
c) con history que fuerza otra lectura (campo arrendado, poco capital, rustico)

Pregunta 2 (vaga, "Y eso en zona nucleo?"):
d) con history que da contexto de feedlot (deberia reescribir a algo
   entendible y recuperar chunks de ganaderia)

Esperado:
- (b) y (c) producen rewrites distintos de (a)
- (d) produce un rewrite con sentido (no devuelve la pregunta vaga)
- Los retrievals cambian segun el rewrite (a veces solo el orden, a
  veces los chunks mismos)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.strategies.query_rewrite import QueryRewriteStrategy

QUERY = "Cuanto cuesta un kilo de novillo en feedlot?"

HISTORY_B = [
    {"role": "user", "content": "Hola, estoy pensando en armar un feedlot"},
    {"role": "assistant", "content": "Buenisimo, el feedlot puede ser rentable si lo planteas bien."},
    {"role": "user", "content": "Que cantidad de cabezas me recomendas empezar?"},
    {"role": "assistant", "content": "Depende del capital, pero arrancar con 200-500 cabezas es razonable."},
    {"role": "user", "content": "Y la zona importa para los costos?"},
    {"role": "assistant", "content": "Si, la zona centro nucleo tiene mejor oferta de insumos y servicios."},
    {"role": "user", "content": "Dale, entonces me inclino por zona nucleo, campo propio."},
    {"role": "user", "content": QUERY},
]

HISTORY_C = [
    {"role": "user", "content": "Mira, tengo campo arrendado en el sur de Buenos Aires, son 200 ha"},
    {"role": "assistant", "content": "Bien, 200 ha es un lindo tamanio para empezar algo."},
    {"role": "user", "content": "El tema es que tengo poco capital, asi que voy a algo chico"},
    {"role": "assistant", "content": "Una opcion es recria a campo con suplementacion, sin grandes inversiones."},
    {"role": "user", "content": "Y si quiero hacer ciclo completo en vez de recria?"},
    {"role": "assistant", "content": "Ciclo completo requiere mas capital: instalaciones, comida, sanidad."},
    {"role": "user", "content": "La idea seria algo rustico, sin meter mucho en instalaciones"},
    {"role": "user", "content": QUERY},
]

# Caso d: pregunta vaga con history que da contexto
QUERY_VAGA = "Y eso en zona nucleo?"
HISTORY_D = [
    {"role": "user", "content": "Hola, estoy evaluando hacer un feedlot"},
    {"role": "assistant", "content": "Dale, te ayudo. Que tipo de feedlot pensas: ciclo completo, recria, o engorde?"},
    {"role": "user", "content": "Ciclo completo, unas 300 cabezas"},
    {"role": "assistant", "content": "OK, en zona nucleo hay buena oferta de maiz para el feedlot. Hay algo que quieras saber?"},
    {"role": "user", "content": "Cuanto me sale mantener una cabeza en feedlot?"},
    {"role": "assistant", "content": "Depende del planteo, pero los costos de feedlot van de 130 a 235 US$/ha..."},
    {"role": "user", "content": "Y los kilos de venta del novillo en feedlot?"},
    {"role": "assistant", "content": "Entre 3,16 y 3,72 US$/kg segun el planteo que uses (pag. 76)."},
    {"role": "user", "content": QUERY_VAGA},
]


def main() -> int:
    s = QueryRewriteStrategy()

    print("=" * 80)
    print(f"QUERY BASE: {QUERY!r}")
    print("=" * 80)

    # Caso a: sin history
    print("\n[CASO A] SIN HISTORY")
    print("-" * 80)
    r_a = s.retrieve(QUERY, history=None)
    print(f"  rewritten: {r_a.extra['rewritten_query']!r}")
    print(f"  changed:   {r_a.extra['rewritten_changed']}")
    print(f"  intent:    {r_a.intent}")
    print(f"  llm in/out: {r_a.llm_input_tokens}/{r_a.llm_output_tokens}")
    print(f"  retrieval: {r_a.retrieval_ms:.0f}ms")
    print(f"  top 3 secciones: {[i.seccion for i in r_a.items[:3]]}")
    print(f"  top 3 paginas:   {[i.pagina for i in r_a.items[:3]]}")
    a_ids = [i.chunk_id for i in r_a.items]

    # Caso b: history neutral
    print("\n[CASO B] HISTORY NEUTRAL (feedlot en zona nucleo, campo propio)")
    print("-" * 80)
    r_b = s.retrieve(QUERY, history=HISTORY_B)
    print(f"  rewritten: {r_b.extra['rewritten_query']!r}")
    print(f"  changed:   {r_b.extra['rewritten_changed']}")
    print(f"  intent:    {r_b.intent}")
    print(f"  llm in/out: {r_b.llm_input_tokens}/{r_b.llm_output_tokens}")
    print(f"  retrieval: {r_b.retrieval_ms:.0f}ms")
    print(f"  top 3 secciones: {[i.seccion for i in r_b.items[:3]]}")
    print(f"  top 3 paginas:   {[i.pagina for i in r_b.items[:3]]}")
    b_ids = [i.chunk_id for i in r_b.items]

    # Caso c: history que fuerza un planteo rustico
    print("\n[CASO C] HISTORY FORZADO (campo arrendado, poco capital, rustico)")
    print("-" * 80)
    r_c = s.retrieve(QUERY, history=HISTORY_C)
    print(f"  rewritten: {r_c.extra['rewritten_query']!r}")
    print(f"  changed:   {r_c.extra['rewritten_changed']}")
    print(f"  intent:    {r_c.intent}")
    print(f"  llm in/out: {r_c.llm_input_tokens}/{r_c.llm_output_tokens}")
    print(f"  retrieval: {r_c.retrieval_ms:.0f}ms")
    print(f"  top 3 secciones: {[i.seccion for i in r_c.items[:3]]}")
    print(f"  top 3 paginas:   {[i.pagina for i in r_c.items[:3]]}")
    c_ids = [i.chunk_id for i in r_c.items]

    # Caso d: pregunta vaga con history
    print("\n[CASO D] PREGUNTA VAGA + history con contexto de feedlot")
    print("-" * 80)
    print(f"  query: {QUERY_VAGA!r}")
    r_d = s.retrieve(QUERY_VAGA, history=HISTORY_D)
    print(f"  rewritten: {r_d.extra['rewritten_query']!r}")
    print(f"  changed:   {r_d.extra['rewritten_changed']}")
    print(f"  intent:    {r_d.intent}")
    print(f"  llm in/out: {r_d.llm_input_tokens}/{r_d.llm_output_tokens}")
    print(f"  retrieval: {r_d.retrieval_ms:.0f}ms")
    print(f"  top 3 secciones: {[i.seccion for i in r_d.items[:3]]}")
    print(f"  top 3 paginas:   {[i.pagina for i in r_d.items[:3]]}")
    d_ids = [i.chunk_id for i in r_d.items]

    # Comparacion
    print("\n" + "=" * 80)
    print("COMPARACION")
    print("=" * 80)

    print(f"\nA vs B: same set? {set(a_ids) == set(b_ids)}")
    print(f"  rewrites diff?   {r_a.extra['rewritten_query'] != r_b.extra['rewritten_query']}")
    print(f"  a==b: {a_ids == b_ids}")
    if set(a_ids) != set(b_ids):
        diff_ab = set(a_ids) ^ set(b_ids)
        print(f"  diff chunks: {len(diff_ab)} de {len(a_ids)}")

    print(f"\nA vs C: same set? {set(a_ids) == set(c_ids)}")
    print(f"  rewrites diff?   {r_a.extra['rewritten_query'] != r_c.extra['rewritten_query']}")
    print(f"  a==c: {a_ids == c_ids}")
    if set(a_ids) != set(c_ids):
        diff_ac = set(a_ids) ^ set(c_ids)
        print(f"  diff chunks: {len(diff_ac)} de {len(a_ids)}")

    print(f"\nB vs C: same set? {set(b_ids) == set(c_ids)}")
    print(f"  rewrites diff?   {r_b.extra['rewritten_query'] != r_c.extra['rewritten_query']}")
    print(f"  b==c: {b_ids == c_ids}")
    if set(b_ids) != set(c_ids):
        diff_bc = set(b_ids) ^ set(c_ids)
        print(f"  diff chunks: {len(diff_bc)} de {len(b_ids)}")

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    a_rew = r_a.extra["rewritten_query"]
    b_rew = r_b.extra["rewritten_query"]
    c_rew = r_c.extra["rewritten_query"]
    print(f"  a == b: {a_rew == b_rew}   |  chunks: {a_ids == b_ids}")
    print(f"  a == c: {a_rew == c_rew}   |  chunks: {a_ids == c_ids}")
    print(f"  b == c: {b_rew == c_rew}   |  chunks: {b_ids == c_ids}")
    if b_rew != c_rew:
        print("\n  >> query_rewrite produce reescrituras DISTINTAS segun history ✓")
    if a_ids != b_ids or a_ids != c_ids:
        print("  >> el retrieval refleja los cambios en la query ✓")
    # Sobre el caso d
    print()
    print(f"  caso d: query vaga {QUERY_VAGA!r}")
    print(f"  rewrite d: {r_d.extra['rewritten_query']!r}")
    if r_d.extra["rewritten_query"] != QUERY_VAGA:
        print(f"  >> el rewrite DE LA PREGUNTA VAGA produjo una query con sentido ✓")
    else:
        print(f"  >> el rewrite NO cambio la pregunta vaga (sin history suficiente)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
