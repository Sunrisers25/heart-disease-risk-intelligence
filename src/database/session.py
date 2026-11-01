"""SQLAlchemy session management and database engine configuration."""

import logging
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DB_PATH
from src.database.models import Base

logger = logging.getLogger("src.database.session")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enforce foreign key constraints on SQLite connections."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception as exc:
        logger.warning("Could not set SQLite foreign_keys PRAGMA: %s", exc)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """Create all application tables in the configured database if not present."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Initialized database tables at %s", DB_PATH)


def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional SQLAlchemy database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

