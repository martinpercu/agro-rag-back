"""DB session dependency for FastAPI."""
from __future__ import annotations

from sqlalchemy.orm import Session

from db.base import SessionLocal


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
