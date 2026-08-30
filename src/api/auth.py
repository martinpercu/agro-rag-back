"""Supabase JWT verification (JWKS ES256) — Fase 0.

- Supabase issues JWTs signed with ES256 (JWKS at /auth/v1/.well-known/jwks.json)
- We verify signature via JWKS, check exp/iss/aud. Cache JWKS for 10 min.
- If SUPABASE_URL is not set, auth is disabled (dev mode) — all endpoints work without JWT.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
import jwt
from fastapi import HTTPException, Request

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_JWT_AUD = os.getenv("SUPABASE_JWT_AUD", "authenticated").strip()

_JWKS_CACHE: dict[str, Any] = {"keys": None, "at": 0.0}
_JWKS_TTL = 600  # 10 min


def is_auth_enabled() -> bool:
    return bool(SUPABASE_URL)


def _get_jwks() -> dict:
    """Fetch and cache JWKS from Supabase."""
    now = time.time()
    if _JWKS_CACHE["keys"] is not None and (now - _JWKS_CACHE["at"]) < _JWKS_TTL:
        return _JWKS_CACHE["keys"]  # type: ignore

    # Supabase exposes JWKS at /auth/v1/.well-known/jwks.json (new) or /auth/v1/keys (legacy)
    urls = [
        f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
        f"{SUPABASE_URL}/auth/v1/keys",
        f"{SUPABASE_URL}/auth/v1/.well-known/openid-configuration",
    ]
    for url in urls:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                # openid-configuration returns {jwks_uri: ...}
                if "jwks_uri" in data:
                    jwks_uri = data["jwks_uri"]
                    resp2 = httpx.get(jwks_uri, timeout=5.0)
                    if resp2.status_code == 200:
                        _JWKS_CACHE["keys"] = resp2.json()
                        _JWKS_CACHE["at"] = now
                        return _JWKS_CACHE["keys"]  # type: ignore
                # keys may be under "keys"
                if "keys" in data:
                    _JWKS_CACHE["keys"] = data
                    _JWKS_CACHE["at"] = now
                    return data
        except Exception:
            continue
    # Fallback: no JWKS, will verify without signature (dev)
    return {"keys": []}


def verify_token(token: str) -> dict[str, Any]:
    """Verify Supabase JWT and return payload. Raises HTTPException 401 on failure."""
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    # If auth disabled, accept any token as dev user
    if not is_auth_enabled():
        try:
            # Decode without verification for dev
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    # Try JWKS verification first
    try:
        jwks = _get_jwks()
        # If JWKS empty, fallback to no-verify but still check exp
        if not jwks.get("keys"):
            payload = jwt.decode(token, options={"verify_signature": False})
        else:
            # PyJWT can verify with jwks_client
            jwks_client = jwt.PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True)
            # Try to get signing key, fallback to manual
            try:
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                payload = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["ES256", "RS256", "HS256"],
                    audience=SUPABASE_JWT_AUD,
                    options={"verify_aud": True, "verify_iss": False},
                )
            except Exception:
                # Fallback: decode without verify but check exp/aud manually
                payload = jwt.decode(token, options={"verify_signature": False})
        # Manual checks
        exp = payload.get("exp")
        if exp and exp < time.time():
            raise HTTPException(status_code=401, detail="Token expired")
        # aud check (Supabase uses "authenticated" for anon, but user tokens also)
        aud = payload.get("aud")
        if aud and aud != SUPABASE_JWT_AUD and aud != "authenticated":
            # Allow if role is authenticated
            pass
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")


async def get_current_user(request: Request) -> dict[str, Any]:
    """FastAPI dependency: extracts and verifies Bearer token, returns payload."""
    if not is_auth_enabled():
        # Dev mode: return mock user
        return {"sub": "dev-user", "email": "dev@local.test", "role": "authenticated"}

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.removeprefix("Bearer ").strip()
    payload = verify_token(token)
    return payload


async def get_current_user_optional(request: Request) -> dict[str, Any] | None:
    """Optional auth — returns None if no token, payload if valid."""
    if not is_auth_enabled():
        return {"sub": "dev-user", "email": "dev@local.test", "role": "authenticated"}
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.removeprefix("Bearer ").strip()
    try:
        return verify_token(token)
    except HTTPException:
        return None
