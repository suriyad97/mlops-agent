"""SQLite persistence: projects, threads, messages, reports, failure memory."""
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    String,
    Text,
    create_engine,
)
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

DB_URL = "sqlite:///platform.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    repo_url: Mapped[str] = mapped_column(String(500))
    pat_encrypted: Mapped[str] = mapped_column(Text)
    profile: Mapped[dict] = mapped_column(JSON, default=dict)   # type, target, metrics, endpoint, drift…
    stage: Mapped[str] = mapped_column(String(40), default="configured")
    local_repo_path: Mapped[str] = mapped_column(String(500), default="")
    base_branch: Mapped[str] = mapped_column(String(200), default="main")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    threads: Mapped[list["Thread"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    reports: Mapped[list["Report"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    failures: Mapped[list["FailureRecord"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Thread(Base):
    __tablename__ = "threads"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    title: Mapped[str] = mapped_column(String(200), default="New thread")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    project: Mapped[Project] = relationship(back_populates="threads")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="thread", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"))
    role: Mapped[str] = mapped_column(String(12))               # user | assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    thread: Mapped[Thread] = relationship(back_populates="messages")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    kind: Mapped[str] = mapped_column(String(40))               # blueprint | capability | gap | validation
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    project: Mapped[Project] = relationship(back_populates="reports")


class FailureRecord(Base):
    __tablename__ = "failure_memory"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    signature: Mapped[str] = mapped_column(Text)
    rca: Mapped[str] = mapped_column(Text, default="")
    fix: Mapped[dict] = mapped_column(JSON, default=dict)
    outcome: Mapped[str] = mapped_column(String(40), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    project: Mapped["Project"] = relationship(back_populates="failures")


def init_db() -> None:
    Base.metadata.create_all(engine)
    # Add columns introduced after initial schema (SQLite ALTER TABLE is additive-only)
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(projects)"))}
        for col, ddl in [
            ("local_repo_path", "ALTER TABLE projects ADD COLUMN local_repo_path VARCHAR(500) DEFAULT ''"),
            ("base_branch",     "ALTER TABLE projects ADD COLUMN base_branch VARCHAR(200) DEFAULT 'main'"),
        ]:
            if col not in existing:
                conn.execute(text(ddl))
        conn.commit()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
