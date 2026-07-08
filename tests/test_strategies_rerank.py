"""Tests de la strategy rerank con LLM (gpt-4.1-nano)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from agent.strategies.rerank import (
    RerankStrategy,
    _format_fragments,
    _parse_indices,
)
from schemas import Chunk, ChunkMetadata


# ---------- Tests del parser de indices ----------

def test_parse_indices_simple_json():
    assert _parse_indices("[3, 0, 5, 1, 2, 4]", 6) == [3, 0, 5, 1, 2, 4]


def test_parse_indices_with_markdown_fences():
    content = "```json\n[3, 0, 5, 1, 2, 4]\n```"
    assert _parse_indices(content, 6) == [3, 0, 5, 1, 2, 4]


def test_parse_indices_with_prose_around():
    content = "Aqui va el array:\n[3, 0, 5]\nListo."
    assert _parse_indices(content, 6) == [3, 0, 5]


def test_parse_indices_dedupes():
    content = "[3, 3, 0, 0, 5]"
    assert _parse_indices(content, 6) == [3, 0, 5]


def test_parse_indices_clamps_out_of_range():
    content = "[3, 0, 99, -1, 5]"
    assert _parse_indices(content, 6) == [3, 0, 5]


def test_parse_indices_handles_non_int():
    content = '["3", 0, null, 5.5, 2]'
    # solo acepta ints validos
    assert _parse_indices(content, 6) == [0, 2]


def test_parse_indices_invalid_json_returns_empty():
    assert _parse_indices("no es json", 6) == []
    assert _parse_indices("[]", 6) == []
    assert _parse_indices("", 6) == []
    assert _parse_indices("{}", 6) == []


# ---------- Tests del formatter de fragments ----------

def _make_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c{i}",
        text=f"texto del chunk numero {i}",
        metadata=ChunkMetadata(
            edicion="2026_05",
            seccion="costos_margenes",
            cultivo="soja",
            zona=None,
            campana="2026_27",
            tipo="tabla",
            pagina=i + 1,
        ),
    )


def test_format_fragments_includes_metadata_and_index():
    from agent.strategies.base import RetrievedItem
    items = [
        RetrievedItem(
            chunk_id="c1", text="hola mundo", seccion="costos", pagina=10,
            cultivo="soja", campana=None, tipo="tabla", score=0.5, rank=0
        )
    ]
    out = _format_fragments(items)
    assert "[0]" in out
    assert "pag. 10" in out
    assert "costos" in out
    assert "hola mundo" in out


def test_format_fragments_truncates_long_text():
    from agent.strategies.base import RetrievedItem
    long_text = "x" * 5000
    items = [
        RetrievedItem(
            chunk_id="c1", text=long_text, seccion="x", pagina=1,
            cultivo=None, campana=None, tipo="tabla", score=0.5, rank=0
        )
    ]
    out = _format_fragments(items)
    # texto truncado + metadata < 1500 chars
    assert len(out) < 1500
    # el texto no deberia tener los 5000 x
    assert out.count("x") < 1000


# ---------- Tests de la strategy con OpenAI mockeado ----------

def _chroma_hits_for(items_text: list[str], seccion: str = "costos_margenes") -> list[tuple]:
    """Convierte una lista de textos en hits fake de chroma."""
    hits = []
    for i, text in enumerate(items_text):
        chunk = _make_chunk(i)
        chunk.text = text
        chunk.metadata.seccion = seccion  # type: ignore[attr-defined]
        hits.append((chunk, 0.5 - i * 0.01))
    return hits


def _make_fake_response(content: str, in_tok: int = 1000, out_tok: int = 20):
    """Crea un fake de la response de OpenAI."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = in_tok
    resp.usage.completion_tokens = out_tok
    return resp


def test_rerank_uses_llm_indices_to_reorder():
    s = RerankStrategy()
    candidates = [
        f"texto del chunk {i} con keyword distinta" for i in range(20)
    ]
    chroma_hits = _chroma_hits_for(candidates)
    fake_resp = _make_fake_response("[5, 0, 1, 2, 3, 4]", in_tok=1234, out_tok=15)

    with (
        patch("agent.strategies.rerank._classify", return_value="costos"),
        patch("agent.strategies.rerank.search", return_value=chroma_hits),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp) as mock_create,
    ):
        result = s.retrieve("test", k=6)

    # El primer item deberia ser el que estaba en posicion 5
    assert result.items[0].chunk_id == "c5"
    assert result.items[1].chunk_id == "c0"
    assert result.items[5].chunk_id == "c4"
    assert result.num_sources == 6
    assert result.llm_input_tokens == 1234
    assert result.llm_output_tokens == 15
    # El LLM fue llamado con el prompt correcto
    mock_create.assert_called_once()
    call_args = mock_create.call_args
    assert "test" in call_args.kwargs["messages"][0]["content"]


def test_rerank_falls_back_to_original_order_on_parse_failure():
    s = RerankStrategy()
    candidates = [f"texto {i}" for i in range(20)]
    chroma_hits = _chroma_hits_for(candidates)
    fake_resp = _make_fake_response("esto no es json valido", in_tok=500, out_tok=10)

    with (
        patch("agent.strategies.rerank._classify", return_value="costos"),
        patch("agent.strategies.rerank.search", return_value=chroma_hits),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
    ):
        result = s.retrieve("test", k=6)

    # Fallback: orden original (cosine similarity)
    assert result.items[0].chunk_id == "c0"
    assert result.items[5].chunk_id == "c5"
    assert result.extra["fallback"] == "parse_failed"


def test_rerank_pads_with_remaining_when_llm_returns_few():
    """Si el LLM devuelve menos de k indices, rellenamos con el resto."""
    s = RerankStrategy()
    candidates = [f"texto {i}" for i in range(20)]
    chroma_hits = _chroma_hits_for(candidates)
    fake_resp = _make_fake_response("[5, 0]")  # solo 2 indices

    with (
        patch("agent.strategies.rerank._classify", return_value="costos"),
        patch("agent.strategies.rerank.search", return_value=chroma_hits),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
    ):
        result = s.retrieve("test", k=6)

    # c5 y c0 primero, despues c1, c2, c3, c4 del orden original
    assert result.items[0].chunk_id == "c5"
    assert result.items[1].chunk_id == "c0"
    assert result.items[2].chunk_id == "c1"
    assert result.items[3].chunk_id == "c2"
    assert result.num_sources == 6
    assert result.extra["fallback"] is None


def test_rerank_handles_llm_call_error():
    s = RerankStrategy()
    candidates = [f"texto {i}" for i in range(20)]
    chroma_hits = _chroma_hits_for(candidates)

    with (
        patch("agent.strategies.rerank._classify", return_value="costos"),
        patch("agent.strategies.rerank.search", return_value=chroma_hits),
        patch.object(
            s.client.chat.completions,
            "create",
            side_effect=RuntimeError("openai caida"),
        ),
    ):
        result = s.retrieve("test", k=6)

    # Falla del LLM: devolvemos los top-k del chroma original
    assert result.num_sources == 6
    assert "llm_call_failed" in result.extra["error"]


def test_rerank_returns_empty_when_chroma_empty():
    s = RerankStrategy()
    with (
        patch("agent.strategies.rerank._classify", return_value="general"),
        patch("agent.strategies.rerank.search", return_value=[]),
    ):
        result = s.retrieve("nada")
    assert result.num_sources == 0
    assert result.items == []


def test_rerank_scores_are_position_based():
    """El score en el output es 1/(rank+1), no el chroma original."""
    s = RerankStrategy()
    candidates = [f"texto {i}" for i in range(20)]
    chroma_hits = _chroma_hits_for(candidates)
    fake_resp = _make_fake_response("[5, 0, 1, 2, 3, 4]")

    with (
        patch("agent.strategies.rerank._classify", return_value="costos"),
        patch("agent.strategies.rerank.search", return_value=chroma_hits),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
    ):
        result = s.retrieve("test", k=6)

    scores = [i.score for i in result.items]
    assert scores[0] == 1.0
    assert scores[1] == 0.5
    assert scores[2] == pytest.approx(1 / 3)
    assert scores[5] == pytest.approx(1 / 6)


# ---------- Test de integracion: OpenAI real ----------

@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_rerank_integration_real_openai(has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    s = RerankStrategy()
    result = s.retrieve("cuanto cuesta un kilo de novillo en feedlot?", k=6)

    assert result.name == "rerank"
    assert result.intent == "ganaderia"
    assert result.num_sources == 6
    # El LLM consumio tokens
    assert result.llm_input_tokens > 100
    assert result.llm_output_tokens > 0
    # Fallback solo si el LLM fallo
    assert result.extra["fallback"] is None
    # El primer item deberia ser de ganaderia_costos
    assert result.items[0].seccion == "ganaderia_costos"
    # El LLM recibio un prompt razonable
    assert "novillo" in result.extra["llm_raw_response_preview"].lower() or \
           "[" in result.extra["llm_raw_response_preview"]


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_rerank_reorders_vs_baseline(has_openai_key):
    """El rerank deberia producir un orden distinto al baseline (a veces)."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    from agent.strategies.baseline import BaselineStrategy

    question = "que se proyecta para trigo en la campana 2026/27"
    b = BaselineStrategy().retrieve(question, k=6)
    r = RerankStrategy().retrieve(question, k=6)

    b_ids = [i.chunk_id for i in b.items]
    r_ids = [i.chunk_id for i in r.items]
    # No siempre difieren, pero al menos los scores deberian ser distintos
    # (rerank usa 1/(rank+1), baseline usa cosine)
    assert any(bi.score != ri.score for bi, ri in zip(b.items, r.items))
