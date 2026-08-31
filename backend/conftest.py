"""Pytest bootstrap for the backend test suite.

This root ``conftest.py`` runs before any test module is imported, so it is the
right place to establish two process-wide preconditions the suite depends on:

1. **Import path.** ``backend/`` is placed on ``sys.path`` so ``import app...``
   resolves whether pytest is launched from ``backend/`` or from the repository
   root. pytest imports the nearest ``conftest.py`` before collecting tests, so
   this happens early enough for every test's ``from app... import`` to work.

2. **Rate limiting disabled for the general suite (Task 14.1).** The production
   app enforces a default 60-requests/minute-per-client limit (Req 18.4). Under
   the test suite a single client (the ``TestClient``, one source address)
   issues far more than 60 requests per minute, so leaving the limiter enabled
   would spuriously turn later requests into ``429``s and break otherwise-valid
   tests. Setting ``RATE_LIMIT_ENABLED=false`` in the environment *before* the
   settings are first loaded wires ``app.main``'s limiter to ``enabled=False``,
   making the rate-limit middleware fully transparent for the general run.

   The dedicated rate-limit test in ``tests/test_security.py`` does **not** rely
   on the global limiter: it builds its own low-limit app instance with the
   limiter explicitly enabled, so it still proves the ``429`` behaviour.

The settings object is ``lru_cache``-d, so on the off chance it was already
materialised (e.g. by an import side effect) its cache is cleared after the
environment is set, guaranteeing the disabled flag takes effect.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 1. Ensure ``backend/`` (this file's directory) is importable regardless of the
#    working directory pytest is invoked from.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# 2. Disable per-client rate limiting for the general suite before settings load.
#    Set unconditionally (not ``setdefault``) so a developer's inherited env var
#    cannot re-enable the limiter mid-suite.
os.environ["RATE_LIMIT_ENABLED"] = "false"

# Clear any already-materialised settings cache so the flag above is honoured
# even if configuration was loaded before this module ran.
try:  # pragma: no cover - defensive; the import path is exercised by every test
    from app.core.config import get_settings

    get_settings.cache_clear()
except Exception:  # noqa: BLE001 - never let bootstrap break collection
    pass
