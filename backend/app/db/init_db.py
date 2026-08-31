"""Table-creation entry point for the Price Truth schema.

Running

    python -m app.db.init_db

provisions all six tables (``products``, ``category_price_stats``,
``price_snapshots``, ``pack_size_history``, ``platform_prices``,
``category_seasonality``) against the database configured by
``DATABASE_URL`` (read via :mod:`app.core.config`, Req 13.1).

This is a deliberately standalone, re-runnable module rather than something
wired into application startup: provisioning schema is an operational step
that should be invoked explicitly (locally, in ``docker-compose``, or as a
deploy hook), not on every process boot. Keeping it separate from
``main.py`` also preserves the data layer's isolation from the API layer
(Req 17.5).

:func:`create_all` is idempotent. ``Base.metadata.create_all`` issues its
DDL with ``checkfirst=True`` by default, so existing tables are skipped and
no data is dropped when the module is run more than once.
"""

from __future__ import annotations

# Importing the models module executes every mapped-class definition, which
# registers all six tables on ``Base.metadata``. Without this import
# ``create_all`` would see an empty metadata collection.
from app.db.models import Base
from app.db.session import engine


def create_all() -> None:
    """Create every registered table on the configured engine (idempotent)."""

    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_all()
