"""API tests for the data-sources disclosure endpoint (Task 13.3).

Exercises ``GET /api/v1/data-sources`` end-to-end through a FastAPI
``TestClient``. The endpoint returns a static disclosure document (it touches
neither the database nor Redis), so these tests verify that the payload carries
every required disclosure:

* an accessible description of the platform's data sources and their known
  limitations (Req 10.2) - the Kaggle Amazon & Flipkart datasets and the Open
  Food Facts public API;
* the crowd-sourced Open Food Facts notice (Req 10.3);
* the category-level / snapshot-data buy-timing disclosure (Req 10.1);
* the transparent weak-supervision labelling caveat; and
* the explicit statement that live scraping of Amazon/Flipkart is not a core
  data source (Req 10.4).

No dependency override is needed because the endpoint reads no request-scoped
state. The disclosure statements are asserted by identity against the module
constants so the test pins the exact contract rather than brittle prose.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.v1.meta import (
    DISCLOSURE_BUY_TIMING_CATEGORY_LEVEL,
    DISCLOSURE_NO_LIVE_SCRAPING,
    DISCLOSURE_OFF_CROWD_SOURCED,
    DISCLOSURE_WEAK_SUPERVISION_LABELS,
)
from app.main import app

client = TestClient(app)


def test_data_sources_returns_sources_and_all_disclosures():
    """The endpoint yields 200 describing sources and all required disclosures.

    Verifies the structured payload (Req 10.2-10.4, 14.4): the three data
    sources are named, and the four disclosure statements are present by
    identity - crowd-sourced OFF (Req 10.3), category-level/snapshot buy-timing
    (Req 10.1), weak-supervision labels, and no live scraping (Req 10.4).
    """
    resp = client.get("/api/v1/data-sources")

    assert resp.status_code == 200
    body = resp.json()

    # Req 10.2: data sources are described. The three expected sources appear.
    sources = body["data_sources"]
    assert isinstance(sources, list) and len(sources) >= 3
    names = " | ".join(source["name"] for source in sources).lower()
    assert "amazon" in names and "kaggle" in names
    assert "flipkart" in names
    assert "open food facts" in names
    # Every source lists at least one known limitation (Req 10.2).
    assert all(source.get("limitations") for source in sources)

    # Req 10.1, 10.3, 10.4 + weak-supervision: each disclosure present by identity.
    disclosures = body["disclosures"]
    assert disclosures["buy_timing_category_level"] == DISCLOSURE_BUY_TIMING_CATEGORY_LEVEL
    assert disclosures["open_food_facts_crowd_sourced"] == DISCLOSURE_OFF_CROWD_SOURCED
    assert (
        disclosures["discount_labels_weak_supervision"]
        == DISCLOSURE_WEAK_SUPERVISION_LABELS
    )
    assert disclosures["no_live_scraping"] == DISCLOSURE_NO_LIVE_SCRAPING

    # The flat limitations list mirrors the structured disclosures.
    assert set(body["limitations"]) == set(disclosures.values())


def test_data_sources_disclosure_wording_covers_required_notices():
    """The disclosure wording explicitly covers the required notices.

    Beyond structural presence, assert the honest-limitations language a
    reviewer looks for: crowd-sourced (Req 10.3), category-level + snapshot
    (Req 10.1), and that scraping is *not* used (Req 10.4).
    """
    resp = client.get("/api/v1/data-sources")

    assert resp.status_code == 200
    disclosures = resp.json()["disclosures"]

    assert "crowd-sourced" in disclosures["open_food_facts_crowd_sourced"].lower()

    buy_timing = disclosures["buy_timing_category_level"].lower()
    assert "category level" in buy_timing or "category-level" in buy_timing
    assert "snapshot" in buy_timing

    no_scraping = disclosures["no_live_scraping"].lower()
    assert "scraping" in no_scraping
    assert "not" in no_scraping
