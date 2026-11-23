from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    workspaces: Mapped[list["Workspace"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Workspace(Base):
    __tablename__ = "workspaces"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    owner: Mapped[User] = relationship(back_populates="workspaces")
    documents: Mapped[list["Document"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")
    eval_runs: Mapped[list["EvalRun"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    num_pages: Mapped[int] = mapped_column(Integer, default=0)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    workspace: Mapped[Workspace] = relationship(back_populates="documents")


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    model: Mapped[str] = mapped_column(String(120))
    summary: Mapped[dict] = mapped_column(JSON)
    results: Mapped[list] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    workspace: Mapped[Workspace] = relationship(back_populates="eval_runs")
