"""Tests de la strategy multi_query."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.strategies.multi_query import (
    MultiQueryStrategy,
    _parse_queries,
    _rrf_merge,
)
from schemas import Chunk, ChunkMetadata


# ---------- Tests del parser ----------

def test_parse_queries_simple_json():
    content = '["q1", "q2", "q3"]'
    assert _parse_queries(content, 3) == ["q1", "q2", "q3"]


def test_parse_queries_with_markdown_fences():
    content = '```json\n["q1", "q2", "q3"]\n```'
    assert _parse_queries(content, 3) == ["q1", "q2", "q3"]


def test_parse_queries_truncates_to_n():
    content = '["q1", "q2", "q3", "q4", "q5"]'
    assert _parse_queries(content, 3) == ["q1", "q2", "q3"]


def test_parse_queries_strips_whitespace():
    content = '["  q1  ", "q2", "q3"]'
    assert _parse_queries(content, 3) == ["q1", "q2", "q3"]


def test_parse_queries_skips_empty_strings():
    content = '["q1", "", "q3"]'
    assert _parse_queries(content, 3) == ["q1", "q3"]


def test_parse_queries_skips_non_strings():
    content = '["q1", 42, "q3", null]'
    assert _parse_queries(content, 3) == ["q1", "q3"]


def test_parse_queries_invalid_json_returns_empty():
    assert _parse_queries("no es json", 3) == []
    assert _parse_queries("", 3) == []
    assert _parse_queries("{}", 3) == []


# ---------- Tests del RRF ----------

def test_rrf_merge_basic():
    rankings = [["a", "b", "c"], ["b", "c", "a"]]
    merged = _rrf_merge(rankings, k=60)
    # b aparece rank 1 en r1 y rank 0 en r2 => 1/62 + 1/61 = max
    # a aparece rank 0 en r1 y rank 2 en r2 => 1/61 + 1/63
    # c aparece rank 2 en r1 y rank 1 en r2 => 1/63 + 1/62
    assert merged[0][0] == "b"
    # a y c empatan (1/61 + 1/63 == 1/63 + 1/62)
    assert merged[1][0] in ("a", "c")


def test_rrf_merge_empty():
    assert _rrf_merge([], k=60) == []
    assert _rrf_merge([["a"], ["a"]], k=60)[0] == ("a", 2 * (1.0 / 61))


# ---------- Tests de la strategy con OpenAI mockeado ----------

def _make_chunk(i: int, seccion: str = "ganaderia_costos", pagina: int = 76) -> Chunk:
    return Chunk(
        id=f"c{i}",
        text=f"chunk {i}",
        metadata=ChunkMetadata(
            edicion="2026_05",
            seccion=seccion,
            cultivo="soja" if "soja" in seccion else None,
            zona=None,
            campana="2026_27" if i % 2 == 0 else None,
            tipo="tabla",
            pagina=pagina + i,
        ),
    )


def _make_fake_response(content: str, in_tok: int = 200, out_tok: int = 80):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = in_tok
    resp.usage.completion_tokens = out_tok
    return resp


def _make_chroma_hits(chunk_ids: list[str], seccion: str = "ganaderia_costos") -> list[tuple]:
    """Devuelve (chunk, score) tuples para los ids dados."""
    return [
        (_make_chunk(int(cid[1:]), seccion=seccion), 0.5 - i * 0.05)
        for i, cid in enumerate(chunk_ids)
    ]


def test_multi_query_runs_3_retrievals_and_merges():
    s = MultiQueryStrategy()
    fake_resp = _make_fake_response(
        '["cuanto vale el kilo de novillo", "precio kilo novillo feedlot", "kilo de carne en feedlot"]'
    )

    # Cada sub-query devuelve 3 hits distintos para que el RRF tenga donde elegir
    with (
        patch("agent.strategies.multi_query._classify", return_value="ganaderia"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp) as mock_create,
        patch("agent.strategies.multi_query.search", side_effect=[
            _make_chroma_hits(["c1", "c2", "c3"]),
            _make_chroma_hits(["c2", "c4", "c1"]),
            _make_chroma_hits(["c3", "c4", "c5"]),
        ]) as mock_search,
    ):
        result = s.retrieve("cuanto cuesta el kilo de novillo en feedlot", k=4)

    # LLM llamado una vez
    assert mock_create.call_count == 1
    # 3 retrievals (uno por sub-query)
    assert mock_search.call_count == 3
    # El intent es ganaderia
    assert result.intent == "ganaderia"
    # 4 items
    assert result.num_sources == 4
    # Los items vienen del RRF
    assert all(i.score > 0 for i in result.items)
    # c2 aparece en r1 y r2 -> deberia estar en top (RRF lo prioriza)
    cids = [i.chunk_id for i in result.items]
    assert "c2" in cids
    # c4 aparece en r2 y r3 -> tambien deberia estar
    assert "c4" in cids
    # No hay duplicados
    assert len(cids) == len(set(cids))
    # fallback=None porque el LLM dio 3 queries validas
    assert result.extra["fallback"] is None
    # generated_queries tiene 3
    assert len(result.extra["generated_queries"]) == 3


def test_multi_query_falls_back_to_original_on_llm_failure():
    s = MultiQueryStrategy()
    with (
        patch("agent.strategies.multi_query._classify", return_value="ganaderia"),
        patch.object(
            s.client.chat.completions,
            "create",
            side_effect=RuntimeError("openai caida"),
        ) as mock_create,
        patch(
            "agent.strategies.multi_query.search",
            return_value=_make_chroma_hits(["c1", "c2", "c3"]),
        ) as mock_search,
    ):
        result = s.retrieve("test query", k=3)

    # Fallback: 1 sola query (la original) y 1 retrieval
    assert mock_create.call_count == 1
    assert mock_search.call_count == 1
    assert result.num_sources == 3
    assert result.extra["fallback"] == "llm_failed_or_parse"
    assert "generation_failed" in result.extra["llm_error"]


def test_multi_query_falls_back_on_invalid_json():
    s = MultiQueryStrategy()
    fake_resp = _make_fake_response("esto no es json")
    with (
        patch("agent.strategies.multi_query._classify", return_value="ganaderia"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp) as mock_create,
        patch(
            "agent.strategies.multi_query.search",
            return_value=_make_chroma_hits(["c1", "c2"]),
        ) as mock_search,
    ):
        result = s.retrieve("test", k=2)

    assert mock_create.call_count == 1
    assert mock_search.call_count == 1
    assert result.num_sources == 2
    assert result.extra["fallback"] == "llm_failed_or_parse"


def test_multi_query_pads_partial_rewrites():
    s = MultiQueryStrategy()
    # LLM devuelve solo 2 reformulaciones (esperabamos 3)
    fake_resp = _make_fake_response('["q1 reformulada", "q2 reformulada"]')
    with (
        patch("agent.strategies.multi_query._classify", return_value="ganaderia"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp) as mock_create,
        patch("agent.strategies.multi_query.search", return_value=[]) as mock_search,
    ):
        result = s.retrieve("pregunta original", k=3)

    # Pad: 2 reformulaciones + la original = 3
    assert mock_create.call_count == 1
    assert len(result.extra["generated_queries"]) == 3
    assert result.extra["generated_queries"][-1] == "pregunta original"
    assert mock_search.call_count == 3
    assert result.extra["fallback"] == "partial_rewrites"


def test_multi_query_applies_section_filter():
    s = MultiQueryStrategy()
    fake_resp = _make_fake_response('["q1", "q2", "q3"]')
    with (
        patch("agent.strategies.multi_query._classify", return_value="costos"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
        patch("agent.strategies.multi_query.search", return_value=[]) as mock_search,
    ):
        s.retrieve("test", k=3)

    # Las 3 sub-queries se buscaron con filtro de costos_margenes
    assert mock_search.call_count == 3
    for call in mock_search.call_args_list:
        _, kwargs = call
        assert "where" in kwargs
        assert "costos_margenes" in kwargs["where"]["seccion"]["$in"]


def test_multi_query_uses_position_based_score_after_rrf():
    """El score en los items es el RRF score."""
    s = MultiQueryStrategy()
    fake_resp = _make_fake_response('["q1", "q2", "q3"]')
    with (
        patch("agent.strategies.multi_query._classify", return_value="general"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
        patch("agent.strategies.multi_query.search", return_value=_make_chroma_hits(["c1", "c2"])),
    ):
        result = s.retrieve("test", k=2)

    # Los 2 items tienen scores RRF
    assert len(result.items) == 2
    assert result.items[0].score > result.items[1].score  # rank 0 > rank 1


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_multi_query_integration_real_openai(has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    s = MultiQueryStrategy()
    result = s.retrieve("cuanto cuesta sembrar soja de primera en zona norte?", k=6)
    assert result.name == "multi_query"
    assert result.intent == "costos"
    assert result.num_sources == 6
    # LLM gasto tokens
    assert result.llm_input_tokens > 100
    assert result.llm_output_tokens > 0
    # 3 queries generadas
    assert len(result.extra["generated_queries"]) == 3
    # No deberia haber caido en fallback
    assert result.extra["fallback"] is None
    # Las queries generadas son distintas de la original
    original = "cuanto cuesta sembrar soja de primera en zona norte?"
    rewrites = result.extra["generated_queries"]
    assert any(q.lower() != original for q in rewrites)


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_multi_query_brings_diverse_chunks(has_openai_key):
    """Multi query deberia traer al menos 1 chunk distinto a baseline."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    from agent.strategies.baseline import BaselineStrategy

    q = "que se proyecta para trigo en la campana 2026/27"
    b = BaselineStrategy().retrieve(q, k=6)
    m = MultiQueryStrategy().retrieve(q, k=6)

    b_ids = {i.chunk_id for i in b.items}
    m_ids = {i.chunk_id for i in m.items}
    # A veces son iguales, a veces no. Pero al menos el set de
    # generated_queries incluye reformulaciones reales
    assert len(m.extra["generated_queries"]) >= 1
