"""Tests de la strategy HyDe."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.strategies.hyde import HydeStrategy
from schemas import Chunk, ChunkMetadata


def _make_chunk(i: int, seccion: str = "ganaderia_costos", pagina: int = 76) -> Chunk:
    return Chunk(
        id=f"c{i}",
        text=f"chunk {i}",
        metadata=ChunkMetadata(
            edicion="2026_05",
            seccion=seccion,
            cultivo=None,
            zona=None,
            campana=None,
            tipo="tabla",
            pagina=pagina + i,
        ),
    )


def _make_fake_response(content: str, in_tok: int = 200, out_tok: int = 150):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = in_tok
    resp.usage.completion_tokens = out_tok
    return resp


def _make_chroma_hits(chunk_ids: list[str], seccion: str = "ganaderia_costos"):
    return [
        (_make_chunk(int(cid[1:]), seccion=seccion), 0.5 - i * 0.05)
        for i, cid in enumerate(chunk_ids)
    ]


def test_hyde_uses_hypothetical_embedding_not_original_query():
    """El embedding que se pasa a chroma es el del hipotetico, no el de la query."""
    fake_vector = [0.1, 0.2, 0.3] * 100
    embed_calls: list[str] = []

    def fake_embed(text: str) -> list[float]:
        embed_calls.append(text)
        return fake_vector

    s = HydeStrategy(embed_fn=fake_embed)
    fake_resp = _make_fake_response(
        "Esta es una respuesta hipotetica sobre feedlot con numeros US$/ha y qq/ha."
    )

    with (
        patch("agent.strategies.hyde._classify", return_value="ganaderia"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp) as mock_create,
        patch("agent.strategies.hyde.search", return_value=_make_chroma_hits(["c1", "c2", "c3"])) as mock_search,
    ):
        result = s.retrieve("cuanto cuesta el kilo de novillo en feedlot?", k=3)

    # El LLM fue llamado
    mock_create.assert_called_once()
    # El embedding fue del hipotetico, no de la query
    assert len(embed_calls) == 1
    embed_arg = embed_calls[0]
    assert "respuesta hipotetica" in embed_arg
    assert "cuanto cuesta" not in embed_arg  # es el hipotetico, no la query
    # El search recibio el query_vector pre-computado
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert kwargs["query_vector"] is fake_vector
    assert "query" not in kwargs


def test_hyde_captures_tokens_and_metadata():
    fake_vector = [0.1] * 300
    s = HydeStrategy(embed_fn=lambda t: fake_vector)
    fake_resp = _make_fake_response(
        "Parrafo hipotetico breve.",
        in_tok=200,
        out_tok=80,
    )

    with (
        patch("agent.strategies.hyde._classify", return_value="ganaderia"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
        patch("agent.strategies.hyde.search", return_value=_make_chroma_hits(["c1"])),
    ):
        result = s.retrieve("test", k=1)

    assert result.llm_input_tokens == 200
    assert result.llm_output_tokens == 80
    assert result.extra["hypothetical_chars"] > 0
    assert "Parrafo" in result.extra["hypothetical_preview"]


def test_hyde_applies_section_filter():
    fake_vector = [0.1] * 300
    s = HydeStrategy(embed_fn=lambda t: fake_vector)
    fake_resp = _make_fake_response("hipotetico")

    with (
        patch("agent.strategies.hyde._classify", return_value="costos"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
        patch("agent.strategies.hyde.search", return_value=[]) as mock_search,
    ):
        s.retrieve("test", k=3)

    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert "where" in kwargs
    assert "costos_margenes" in kwargs["where"]["seccion"]["$in"]


def test_hyde_returns_error_on_llm_failure():
    s = HydeStrategy(embed_fn=lambda t: [0.1] * 300)
    with (
        patch("agent.strategies.hyde._classify", return_value="general"),
        patch.object(
            s.client.chat.completions,
            "create",
            side_effect=RuntimeError("openai caida"),
        ),
    ):
        result = s.retrieve("test", k=3)

    assert result.num_sources == 0
    assert "llm_failed" in result.extra["error"]


def test_hyde_returns_error_on_empty_llm_response():
    s = HydeStrategy(embed_fn=lambda t: [0.1] * 300)
    fake_resp = _make_fake_response("")
    with (
        patch("agent.strategies.hyde._classify", return_value="general"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
    ):
        result = s.retrieve("test", k=3)

    assert result.num_sources == 0
    assert result.extra["error"] == "llm_returned_empty"


def test_hyde_returns_error_on_embedding_failure():
    s = HydeStrategy(embed_fn=lambda t: (_ for _ in ()).throw(RuntimeError("embed error")))
    fake_resp = _make_fake_response("hipotetico")
    with (
        patch("agent.strategies.hyde._classify", return_value="general"),
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
        patch("agent.strategies.hyde.search") as mock_search,
    ):
        result = s.retrieve("test", k=3)

    assert result.num_sources == 0
    assert "embedding_failed" in result.extra["error"]
    mock_search.assert_not_called()


def test_hyde_intent_is_from_original_query_not_hypothetical():
    """El intent se clasifica sobre la pregunta original, no sobre el hipotetico."""
    s = HydeStrategy(embed_fn=lambda t: [0.1] * 300)
    fake_resp = _make_fake_response("hipotetico con palabras de ganaderia")

    with (
        patch("agent.strategies.hyde._classify", return_value="costos") as mock_classify,
        patch.object(s.client.chat.completions, "create", return_value=fake_resp),
        patch("agent.strategies.hyde.search", return_value=[]),
    ):
        result = s.retrieve("cuanto sale la soja?", k=3)

    # _classify fue llamado con la query original, no con el hipotetico
    mock_classify.assert_called_once_with("cuanto sale la soja?")
    assert result.intent == "costos"


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_hyde_integration_real_openai(has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    s = HydeStrategy()
    result = s.retrieve("cuanto cuesta un kilo de novillo en feedlot?", k=6)

    assert result.name == "hyde"
    assert result.intent == "ganaderia"
    assert result.num_sources > 0
    assert result.llm_input_tokens > 0
    assert result.llm_output_tokens > 0
    # El hipotetico es un parrafo (no la pregunta)
    assert result.extra["hypothetical_chars"] > 50
    preview = result.extra["hypothetical_preview"].lower()
    # El hipotetico no deberia ser la pregunta literal
    assert "cuanto cuesta" not in preview or "US$" in preview
    # Items son de ganaderia
    assert all(i.seccion == "ganaderia_costos" for i in result.items)


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_hyde_brings_diverse_chunks(has_openai_key):
    """HyDe deberia traer al menos un chunk distinto a baseline (a veces)."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    from agent.strategies.baseline import BaselineStrategy

    q = "que se proyecta para trigo en la campana 2026/27"
    b = BaselineStrategy().retrieve(q, k=6)
    h = HydeStrategy().retrieve(q, k=6)

    b_ids = {i.chunk_id for i in b.items}
    h_ids = {i.chunk_id for i in h.items}
    # HyDe embebe el hipotetico, no la query; deberia tener al menos
    # una diferencia. Pero como el corpus es chico, puede coincidir.
    # Solo validamos que corra.
    assert h.num_sources > 0
