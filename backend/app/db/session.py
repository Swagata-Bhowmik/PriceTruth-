"""SQLAlchemy engine, session factory, and request-scoped session dependency.

This module owns the database connection plumbing only. SQLAlchemy models
live in ``app.db.models`` (a later task) and query helpers in
``app.db.repositories``; nothing here imports them, keeping the data layer's
concerns separately modifiable (Req 17.5).

The connection URL is read exclusively from
:func:`app.core.config.get_settings` so that no endpoint is hardcoded
(Req 13.1). Creating the engine does **not** open a connection:
:func:`sqlalchemy.create_engine` only prepares a lazily-initialised
connection pool, so importing this module never requires a live database.
Connections are established on first use, and the ``/health`` endpoint and
the DB-unreachable path (Req 16.4) rely on that first use surfacing any
connectivity failure cleanly.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

# The engine holds the connection pool. Its construction is lazy with respect
# to the network: no socket is opened until a session actually executes a
# statement. ``pool_pre_ping`` transparently recycles connections dropped by
# the free-tier database between requests.
engine: Engine = create_engine(
    get_settings().DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

# Session factory. ``expire_on_commit=False`` keeps attributes usable after a
# commit, which suits read-mostly request handlers.
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and guarantee it is closed.

    Intended for use as a FastAPI dependency (``Depends(get_db)``): the
    session is provided to the request handler and closed in the ``finally``
    block once the response has been produced, even if the handler raises.
    """

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
