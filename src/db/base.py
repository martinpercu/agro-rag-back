"""SQLAlchemy Base + engine/session helpers for Railway Postgres."""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "")


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        # Fallback for local alembic without env (uses alembic.ini)
        return "postgresql://postgres:postgres@localhost:5432/railway"
    # Railway gives postgresql:// — SQLAlchemy wants postgresql+psycopg://
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
