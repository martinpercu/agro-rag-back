"""Langfuse helpers — local-only 3003, no OTel.

Uses Langfuse SDK >=2.80 directly (manual spans) to avoid
`Connection reset by peer` from OTel exporter
`http://localhost:3003/api/public/otel/v1/traces`.

Pattern:
    from observability import get_langfuse, start_as_current_observation, flush
    from langfuse import propagate_attributes  # for session_id=user_id

    # In handler (trace root) — set session_id so all child spans inherit:
    from langfuse import propagate_attributes
    lf = get_langfuse()
    if lf is not None:
        with propagate_attributes(session_id=user_id, user_id=user_id):
            with lf.start_as_current_observation(name="plan_parse", as_type="span", input={...}) as span:
                ...
                lf.update_current_span(output={...})
        lf.flush()
    else:
        # no-op when LANGFUSE_HOST not set (prod)
        ...

For graph nodes (classifier→...→answerer), each node wraps its logic:
    from observability import get_langfuse
    lf = get_langfuse()
    if lf is not None:
        with lf.start_as_current_observation(name="classifier", as_type="span", input={...}) as obs:
            result = _classify(...)
            lf.update_current_span(output={"intent": result})
    # retriever → as_type="retriever"
    # answerer → as_type="generation", model=..., usage_details={...}

Helper `start_as_current_observation` here is a thin wrapper that also
supports `session_id`/`user_id` via `propagate_attributes` in one call,
so callers can do:
    from observability import start_as_current_observation
    with start_as_current_observation(name="classifier", as_type="span", input=..., session_id=user_id):
        ...
"""
from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any, Generator

_client = None  # type: ignore
_enabled: bool | None = None


def _is_enabled() -> bool:
    global _enabled
    if _enabled is not None:
        return _enabled
    host = os.getenv("LANGFUSE_HOST", "").strip()
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    _enabled = bool(host and pk and sk)
    return _enabled


def get_langfuse():
    """Return Langfuse client or None (disabled/no keys)."""
    global _client
    if _client is not None:
        return _client if _is_enabled() else None
    if not _is_enabled():
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        _client = Langfuse()
        # auth_check validates keys without raising
        try:
            _client.auth_check()
        except Exception:
            pass
        return _client
    except Exception:
        return None


def is_enabled() -> bool:
    return _is_enabled() and get_langfuse() is not None


def flush() -> None:
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass


@contextmanager
def start_as_current_observation(
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    output: Any | None = None,
    metadata: dict | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    usage_details: dict | None = None,
    **kwargs: Any,
) -> Generator[Any, None, None]:
    """Wrapper around Langfuse.start_as_current_observation with session_id support.

    - If Langfuse disabled, yields None (no-op).
    - If session_id/user_id given, wraps in `propagate_attributes` so the observation
      and all children get those trace attributes (user wants session_id=user_id).
    - For generation/retriever, pass as_type accordingly; for answerer add model/usage_details.
    """
    lf = get_langfuse()
    if lf is None:
        yield None
        return

    # If session/user provided, propagate so trace gets session_id
    if session_id or user_id:
        try:
            from langfuse import propagate_attributes  # type: ignore
        except Exception:
            propagate_attributes = nullcontext  # type: ignore

        # propagate_attributes is itself a contextmanager
        try:
            with propagate_attributes(session_id=session_id, user_id=user_id):  # type: ignore
                with lf.start_as_current_observation(  # type: ignore
                    name=name,
                    as_type=as_type,  # type: ignore
                    input=input,
                    output=output,
                    metadata=metadata,
                    model=model,
                    usage_details=usage_details,
                    **kwargs,
                ) as obs:
                    yield obs
        except Exception:
            # Fallback without propagate_attributes
            with lf.start_as_current_observation(  # type: ignore
                name=name,
                as_type=as_type,  # type: ignore
                input=input,
                output=output,
                metadata=metadata,
                model=model,
                usage_details=usage_details,
                **kwargs,
            ) as obs:
                yield obs
        return

    # No session propagation needed
    with lf.start_as_current_observation(  # type: ignore
        name=name,
        as_type=as_type,  # type: ignore
        input=input,
        output=output,
        metadata=metadata,
        model=model,
        usage_details=usage_details,
        **kwargs,
    ) as obs:
        yield obs


def update_current_observation(
    output: Any | None = None,
    metadata: dict | None = None,
    **kwargs: Any,
) -> None:
    """Update current span (generic)."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.update_current_span(output=output, metadata=metadata, **kwargs)  # type: ignore
    except Exception:
        pass


def update_current_generation(
    output: Any | None = None,
    usage_details: dict | None = None,
    model: str | None = None,
    metadata: dict | None = None,
    **kwargs: Any,
) -> None:
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.update_current_generation(  # type: ignore
            output=output,
            usage_details=usage_details,
            model=model,
            metadata=metadata,
            **kwargs,
        )
    except Exception:
        pass
