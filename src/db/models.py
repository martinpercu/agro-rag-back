"""DB models — open schema (location, profile_hints flexible via JSONB)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from db.base import Base


def _uuid():
    return uuid.uuid4()


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    supabase_user_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    email = Column(String, nullable=True)
    role = Column(String, default="user")
    allow_aggregated_use = Column(Boolean, default=False)
    profile_hints = Column(JSONB, default=dict)  # {cultivo_favorito, tipo, campos, ...}
    created_at = Column(DateTime(timezone=True), default=_now)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)

    session = relationship("Session", back_populates="messages")


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    edition_id = Column(String, nullable=True)  # e.g. 2026_05
    query = Column(Text, nullable=True)
    divisions = Column(JSONB, default=list)
    location = Column(JSONB, default=dict)  # {provincia?, departamento?, localidad?, lat?, lng?, label?}
    price_variants = Column(JSONB, default=list)  # [500,520,540]
    client_hint = Column(Text, nullable=True)
    meta = Column("metadata", JSONB, default=dict)  # open bag for future signals
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class Plan(Base):
    __tablename__ = "plans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    investigation_id = Column(UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="SET NULL"), nullable=True)
    edition_id = Column(String, nullable=True)
    total_hectares = Column(String, nullable=True)  # keep as text to avoid float issues, BigNumber in app
    season = Column(String, nullable=True)
    divisions = Column(JSONB, default=list)
    location = Column(JSONB, default=dict)
    price_variants = Column(JSONB, default=list)
    client_hint = Column(Text, nullable=True)
    meta = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class Edition(Base):
    __tablename__ = "editions"

    id = Column(String, primary_key=True)  # 2026_05
    pdf_path = Column(String, nullable=True)
    pages = Column(String, nullable=True)
    chunks = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
