# Implementation Plan: Price Truth Platform

## Overview

This plan converts the design into an incremental, test-driven sequence of coding steps built on top of the existing Phase 0 scaffolding (`backend/app/{api,core,db,ml,services}`, `main.py`, `requirements.txt`, `frontend/package.json`, `docker-compose.yml`, initialized git repo). Nothing in Phase 0 is recreated; each task extends what already exists.

The build is bottom-up so every step is runnable, verifiable, and committable on its own: configuration and error handling first, then the data layer, then the offline data/ML pipeline, then one feature service at a time (each with its endpoint and property/unit tests), then the composite dashboard, cross-cutting security, the React frontend, testing consolidation, and finally deployment configuration.

Key constraints reflected throughout:
- **Snapshot data** — the discount model consumes **category price statistics** (not per-product history) and buy-timing is **category-level**.
- **Zero budget / no manual data collection** — ingestion is automated from Kaggle CSVs + the Open Food Facts API. A small **synthetic fixture** is created early so the pipeline, ML, and services are fully testable without waiting on the real Kaggle download.
- Backend: Python + FastAPI. Frontend: React 18 + Tailwind (Vite). ML: XGBoost + SHAP + Prophet/statsmodels.

Tasks marked with `*` are optional test sub-tasks (property, unit, integration, contract). Every property test cites the design's Correctness Property number and the requirement clause it validates. Each task is scoped small enough to finish and commit on its own to support continuous incremental commits.

## Tasks

- [ ] 1. Backend configuration and error-handling foundation
  - [x] 1.1 Add dependencies and implement the settings module
    - Add `hypothesis`, `slowapi` (rate limiting), and confirm `pytest`, `pytest-cov`, `httpx` are present in `backend/requirements.txt`
    - Create `app/core/config.py` using `pydantic-settings` to read `DATABASE_URL`, `REDIS_URL`, `OFF_BASE_URL`, `OFF_VERSION`, and `CORS_ALLOWED_ORIGIN` from environment variables (no hardcoded endpoints or secrets)
    - _Requirements: 13.1, 13.4, 18.6_

  - [x] 1.2 Implement the structured error payload and central exception handlers
    - Create `app/core/errors.py` with an `ErrorPayload` model producing `{error: {code, message, status, details}}`
    - Register a central exception handler in `main.py` that converts validation errors, domain errors, and unhandled errors into the single payload shape; add a DB-unreachable path returning a 503 with a retry message
    - _Requirements: 15.3, 16.4_

  - [x]* 1.3 Write property test for the structured error payload
    - **Property 23: All error responses use the structured payload**
    - **Validates: Requirements 15.3**

  - [x] 1.4 Configure structured logging and implement DB session + Redis client
    - Create `app/core/logging.py` (structured logging) used to record validation rejections
    - Create `app/db/session.py` (engine, session factory, `get_db` dependency) and `app/db/redis_client.py` (get/set with TTL), both reading connection details from settings
    - _Requirements: 15.4, 16.4, 17.5_

  - [ ] 1.5 Implement the health endpoint
    - Create `app/api/v1/meta.py` with `GET /health` that checks DB and Redis connectivity and returns a success status when both are operational; register the router in `main.py`
    - _Requirements: 16.1, 16.4_

  - [ ]* 1.6 Write unit tests for the health endpoint and exception handlers
    - Cover healthy path, DB-down 503 with retry message, and error-payload shape for a raised domain error
    - _Requirements: 15.3, 16.1, 16.4_

- [ ] 2. Database models and repositories
  - [x] 2.1 Implement SQLAlchemy 2.0 models for the six tables
    - In `app/db/models.py`, define `products`, `category_price_stats`, `price_snapshots`, `pack_size_history`, `platform_prices`, `category_seasonality` per the ER diagram, including the nullable `platform_prices.genuineness_score` and the attribution columns on `pack_size_history`
    - _Requirements: 17.5_

  - [ ] 2.2 Implement repository helpers and table creation
    - In `app/db/repositories.py`, add parameter-bound query helpers (no string concatenation of user input) for reading products, category stats, pack-size history, platform prices, and seasonality
    - Add a `create_all`/lightweight migration entry point wired into local startup and docker-compose
    - _Requirements: 17.5, 18.2_

  - [ ]* 2.3 Write unit tests for models and repositories
    - Verify each repository uses parameter binding and returns expected shapes against a temporary database
    - _Requirements: 18.2_

- [ ] 3. Offline data ingestion pipeline
  - [x] 3.1 Create synthetic fixture datasets
    - Add small synthetic CSVs under `data/raw/fixtures/` matching the Kaggle Amazon/Flipkart schema (price, reference price, discount, rating, rating count, category, pack quantity) plus a few curated `pack_size_history` rows, so the whole pipeline runs without the real download
    - _Requirements: 17.1_

  - [ ] 3.2 Implement CSV load and cleaning into core tables
    - Create `data/scripts/ingest.py` that loads CSVs (pandas), strips currency symbols/thousands separators, coerces numeric fields, drops non-positive prices, clamps discount pct to [0,100], normalizes category labels, and populates `products`, `price_snapshots`, and `platform_prices`
    - Default the input path to the synthetic fixtures; accept a path argument for real data
    - _Requirements: 9.5, 17.1_

  - [ ] 3.3 Compute and persist category price statistics
    - Extend ingestion to reduce ingested rows into per-category distribution stats (mean/median/std/p25/p75 price, mean/std discount pct, rating norms, sample size) and populate `category_price_stats`
    - _Requirements: 2.3_

  - [ ] 3.4 Compute and persist category seasonality
    - Derive a per-category monthly `relative_price_index`, mark the best window, and map the Indian sale calendar (Big Billion Days, Republic Day Sale, Diwali, Prime Day) into `category_seasonality.sale_event`
    - _Requirements: 6.2, 6.5_

  - [ ] 3.5 Finalize the Kaggle download script and full-data run
    - Complete and test `data/scripts/download_kaggle_datasets.py` to fetch the Amazon/Flipkart CSVs into `data/raw/`, then run `ingest.py` against the real files
    - **Depends on the user providing Kaggle API credentials and downloading the CSVs when prompted; the synthetic fixtures keep every downstream task runnable until then**
    - _Requirements: 13.4, 17.1_

  - [ ]* 3.6 Write unit tests for cleaning and statistics computation
    - Test price parsing, discount clamping, invalid-row dropping, and category-stat correctness on the fixtures
    - _Requirements: 2.3, 9.5_

- [ ] 4. Machine learning pipeline
  - [ ] 4.1 Implement snapshot-aware feature engineering
    - In `app/ml/discount_model.py`, implement the feature transform from `(displayed_price, reference_price, category_stats)` into the documented features (`claimed_discount_pct`, `discount_vs_category_z`, `displayed_price_z`, `reference_price_z`, `displayed_vs_median`, `reference_vs_p75`, and review-signal features)
    - _Requirements: 2.3_

  - [ ] 4.2 Implement the transparent weak-supervision labeling rule
    - Add labeling that marks a row `inflated` when the reference price is a category-distribution outlier while the discounted price sits near the norm, and `genuine` otherwise; keep the rule in one documented function for disclosure
    - _Requirements: 2.3, 10.1_

  - [ ] 4.3 Implement the XGBoost training script and model persistence
    - Create `data/scripts/train_discount_model.py` that engineers features, applies labels, trains the XGBoost binary classifier, and serializes it to `data/models/discount_model.pkl` (joblib)
    - _Requirements: 2.3_

  - [ ] 4.4 Implement the load-once inference module
    - Load the serialized model into FastAPI app state at startup (in `main.py`) and expose a reusable `predict_proba` in `app/ml/discount_model.py`; ensure the model is loaded exactly once per process
    - _Requirements: 2.3, 11.2, 12.4_

  - [ ] 4.5 Implement the SHAP explainer with plain-language labels
    - In `app/ml/explainer.py`, build a single `shap.TreeExplainer` from the same loaded model instance; return base value and per-feature contributions in margin space
    - Add `app/ml/feature_labels.py` mapping raw feature names to plain-language labels
    - _Requirements: 3.1, 3.4, 3.5_

  - [ ]* 4.6 Write unit tests for feature engineering, labeling, and load-once behavior
    - Verify feature values on known inputs, labeling on synthetic outliers, and that the model/explainer are instantiated once and reused
    - _Requirements: 2.3, 3.4, 12.4_

- [ ] 5. Checkpoint - backend foundation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Unit Price Comparator service and endpoint
  - [x] 6.1 Implement the unit price comparator service
    - In `app/services/unit_price_service.py`, convert each variant's pack quantity to a common standard unit (g/ml; kg to g and l to ml multiply by 1000), compute unit price = price / standardized quantity, mark the lowest as best value, and exclude variants with missing/non-positive quantity into an `excluded` list with a reason
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x]* 6.2 Write property tests for the unit price comparator
    - **Property 12: Unit price identity per variant** - _Validates: Requirements 5.1_
    - **Property 13: Best value is the minimum unit price** - _Validates: Requirements 5.2_
    - **Property 14: Unit-price comparison is invariant to unit scale** - _Validates: Requirements 5.4_
    - **Property 15: Invalid-quantity variants are excluded, valid ones included once** - _Validates: Requirements 5.5_

  - [ ] 6.3 Implement the unit-price compare endpoint and schemas
    - Add `POST /api/v1/unit-price/compare` in `app/api/v1/unit_price.py` with the `VariantIn` Pydantic model (unit pattern `^(g|kg|ml|l)$`); register the router in `main.py`
    - _Requirements: 5.3, 14.4, 18.1_

  - [ ]* 6.4 Write unit and API tests for the comparator endpoint
    - Cover the example from the design, mixed-unit conversion, and the excluded-variant response shape
    - _Requirements: 5.3, 5.5, 15.3_

- [ ] 7. Data Service (Open Food Facts client, caching, validation)
  - [ ] 7.1 Implement the OFF client with timeout and bounded retries
    - In `app/services/data_service.py`, call `GET {OFF_BASE_URL}/api/{OFF_VERSION}/product/{barcode}.json` via `httpx.AsyncClient` with a custom `User-Agent`, a 5-second timeout, and at most 2 retries before returning a data-unavailable status
    - _Requirements: 9.2, 15.2_

  - [ ] 7.2 Implement the Redis caching layer for OFF and results
    - Cache validated OFF products under `off:product:{barcode}` (24h) and add cache get/set for the other documented keys; on OFF failure return a cached value when a cache hit exists, else data-unavailable
    - _Requirements: 9.2, 9.4, 12.3_

  - [ ] 7.3 Implement external-value validation and missing-field handling
    - Validate every returned value against expected type/range before it reaches a feature module; return present fields and mark missing fields unavailable rather than failing; log rejections
    - Flag OFF-derived results as crowd-sourced/possibly incomplete
    - _Requirements: 9.1, 9.5, 10.3, 15.4, 18.1_

  - [ ]* 7.4 Write property tests for the data service
    - **Property 20: Missing OFF fields degrade gracefully** - _Validates: Requirements 9.1_
    - **Property 21: Cache returns results identical to fresh computation** - _Validates: Requirements 9.4, 12.3_
    - **Property 22: External and input values are validated before use** - _Validates: Requirements 9.5, 15.4, 18.1_
    - **Property 24: OFF-derived results disclose their crowd-sourced origin** - _Validates: Requirements 10.3_

  - [ ]* 7.5 Write unit tests for retries, timeout, and cache fallback
    - Cover timeout to retry (<=2) to data-unavailable, cache-hit-on-failure, and data-unavailable propagation to a consuming module
    - _Requirements: 9.2, 9.3, 15.2_

- [ ] 8. True Discount Checker and SHAP explainability
  - [ ] 8.1 Implement the discount scoring and banding service
    - In `app/services/discount_service.py`, read `category_price_stats`, engineer features, call inference, map `p(genuine)` to `genuineness_score = round(p*100)` in [0,100], assign bands (>=90 genuine, 60-89 moderate, <60 likely_inflated), and compute effective discount pct
    - Handle pre-conditions: missing/<=displayed reference to not-evaluable with reason; missing category stats to limited-verification result with price context and no score
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 8.2 Write property tests for discount scoring, bands, identity, and pre-conditions
    - **Property 2: Genuineness score is always within range** - _Validates: Requirements 2.1_
    - **Property 3: Discount band is a correct total function of the score** - _Validates: Requirements 2.2_
    - **Property 4: Effective discount percentage identity** - _Validates: Requirements 2.4_
    - **Property 5: A discount cannot be evaluated without a valid reference** - _Validates: Requirements 2.5_

  - [ ] 8.3 Implement the discount-check endpoint with the SHAP breakdown
    - Add `POST /api/v1/discount-check` in `app/api/v1/discount.py` returning displayed/reference price, effective discount, score, classification, and the SHAP `explanation` (base value, final score, plain-language contributions with direction); register the router
    - Cache under `discount:{category}:{displayed}:{reference}`
    - _Requirements: 2.4, 3.1, 3.2, 3.3, 3.5, 11.3, 18.1_

  - [ ]* 8.4 Write property tests for the SHAP breakdown
    - **Property 6: SHAP breakdown is complete and plainly labelled** - _Validates: Requirements 3.1, 3.5_
    - **Property 7: SHAP contribution direction matches its sign** - _Validates: Requirements 3.2_
    - **Property 8: SHAP contributions reconcile to the result** - _Validates: Requirements 3.3_

  - [ ]* 8.5 Write unit/API tests for limited verification and not-evaluable paths
    - Cover the 200 limited-verification body (stats missing), the 422 not-evaluable body (bad reference), and boundary scores 59/60/89/90
    - _Requirements: 2.2, 2.5, 2.6_

- [ ] 9. Shrinkflation Timeline service and endpoint
  - [ ] 9.1 Implement the shrinkflation service
    - In `app/services/shrinkflation_service.py`, read `pack_size_history`, return points in chronological order with pack quantity, selling price, computed unit price, and source attribution; compute total percentage change in pack quantity and in unit price when >=2 points exist; return an unavailable message when there is no history
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 9.2 Write property tests for the shrinkflation timeline
    - **Property 9: Shrinkflation timeline is ordered and attributed** - _Validates: Requirements 4.1, 4.4_
    - **Property 10: Unit price identity at each timeline point** - _Validates: Requirements 4.2_
    - **Property 11: Total pack-size and unit-price change identity** - _Validates: Requirements 4.3_

  - [ ] 9.3 Implement the shrinkflation endpoint
    - Add `GET /api/v1/shrinkflation/{product_id}` in `app/api/v1/shrinkflation.py`; register the router
    - _Requirements: 4.1, 4.4, 14.4_

  - [ ]* 9.4 Write unit tests for the shrinkflation endpoint
    - Cover the no-history unavailable message and attribution presence on OFF-sourced vs cited-record points
    - _Requirements: 4.4, 4.5_

- [ ] 10. Cross-Platform Aggregator service and endpoint
  - [ ] 10.1 Implement the cross-platform service
    - In `app/services/cross_platform_service.py`, read `platform_prices`, return the price on each Supported Platform that has data, mark the lowest as best deal when >=2 exist, include each entry's product link, and include a genuineness score only when the listing has one; single-platform to no-comparison message; no platform to unavailable message
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 10.2 Write property tests for the cross-platform aggregator
    - **Property 18: Cross-platform entries mirror available data** - _Validates: Requirements 7.1, 7.3, 7.4_
    - **Property 19: Best deal is the minimum platform price** - _Validates: Requirements 7.2_

  - [ ] 10.3 Implement the cross-platform endpoint
    - Add `GET /api/v1/cross-platform/{product_id}` in `app/api/v1/cross_platform.py`; cache under `crossplatform:{product_id}`; register the router
    - _Requirements: 7.1, 7.3, 14.4_

  - [ ]* 10.4 Write unit tests for the cross-platform endpoint
    - Cover single-platform no-comparison (7.5) and no-platform unavailable (7.6) responses
    - _Requirements: 7.5, 7.6_

- [ ] 11. Buy Timing Signal service and endpoint
  - [ ] 11.1 Implement the seasonality module and buy-timing service
    - In `app/ml/seasonality.py`, build/read the category seasonal profile (Prophet/statsmodels fit where a monthly index exists, else the Indian sale-calendar prior); in `app/services/buy_timing_service.py`, return `buy_now`/`wait`, attach the deepest-discount window on `wait`, always attach the category-level + snapshot-data disclosure statement, and return unavailable when no profile exists
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 10.1_

  - [ ]* 11.2 Write property tests for buy-timing
    - **Property 16: Buy-timing output is category-level, bounded, and disclosed** - _Validates: Requirements 6.1, 6.3, 6.4, 10.1_
    - **Property 17: A "wait" recommendation points to the deepest-discount window** - _Validates: Requirements 6.2_

  - [ ] 11.3 Implement the buy-timing endpoint
    - Add `GET /api/v1/buy-timing/{category}` in `app/api/v1/buy_timing.py`; register the router
    - _Requirements: 6.1, 6.4, 14.4_

  - [ ]* 11.4 Write unit tests for buy-timing
    - Cover the no-seasonality unavailable message and that the sale calendar contains the four named events
    - _Requirements: 6.5, 6.6_

- [ ] 12. Product Search service and endpoints
  - [ ] 12.1 Implement the search service and manual entry
    - In `app/services/search_service.py`, match `products.normalized_name` via Postgres trigram/`ILIKE`, returning name/brand/category per match; empty query to prompt message; zero matches to no-results message + manual-entry affordance; expose a `SelectedProduct` accepted by every feature module; accept manual entry (name, displayed price, reference price, pack quantity) through the same validation
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 12.2 Write property test for search result shaping
    - **Property 1: Search results always carry identifying fields** - _Validates: Requirements 1.2_

  - [ ] 12.3 Implement the search and manual-entry endpoints
    - Add `GET /api/v1/search?q=` and `POST /api/v1/manual-entry` in `app/api/v1/search.py`; cache search results under `search:{sha1(query)}`; register the router
    - _Requirements: 1.1, 1.5, 1.6, 14.4_

  - [ ]* 12.4 Write unit/API tests for search
    - Cover empty-query prompt, no-results + manual-entry message, and the 3-second result contract
    - _Requirements: 1.1, 1.4, 1.5_

- [ ] 13. Composite Dashboard and data-sources endpoints
  - [ ] 13.1 Implement the composite dashboard endpoint with per-module containment
    - Add `GET /api/v1/dashboard/{product_id}` in `app/api/v1/dashboard.py` that calls each feature service independently, wrapping each in try/except so one failing module returns its unavailable payload while the others succeed; register the router
    - _Requirements: 8.1, 8.5, 15.1_

  - [ ] 13.2 Implement the data-sources disclosure endpoint
    - Add `GET /api/v1/data-sources` in `app/api/v1/meta.py` describing data sources and known limitations, the crowd-sourced OFF notice, the category-level/snapshot disclosure, and the statement that live Amazon/Flipkart scraping is not a core data source
    - _Requirements: 10.2, 10.3, 10.4_

  - [ ]* 13.3 Write unit/API tests for containment and disclosure
    - Verify a thrown module error is contained (others still return) and the data-sources payload contains all required disclosures
    - _Requirements: 8.5, 10.2, 10.4, 15.1_

- [ ] 14. Security hardening (cross-cutting)
  - [ ] 14.1 Add request rate limiting
    - Wire a `slowapi` limiter into `main.py` enforcing 60 requests/min/client, returning a 429 rate-limit status on excess
    - _Requirements: 18.4_

  - [ ] 14.2 Tighten CORS and confirm boundary input validation
    - Change `main.py` CORS from `allow_origins=["*"]` to the single configured frontend origin from settings; confirm every endpoint validates input type/length/format at the Pydantic boundary
    - _Requirements: 18.1, 18.3_

  - [ ]* 14.3 Write API tests for security controls
    - Cover rate-limit trigger at 61 req/min, allowed vs disallowed CORS origin, injection strings treated as data (parameterized queries), and absence of secrets in responses
    - _Requirements: 18.2, 18.3, 18.4, 18.6_

- [ ] 15. Checkpoint - backend complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Frontend foundation
  - [ ] 16.1 Set up Vite + React 18 + Tailwind and install dependencies
    - Initialize the Vite React app in `frontend/`, wire Tailwind, install the listed dependencies plus dev dependency `fast-check`, and configure the three responsive breakpoints (<=480 single column, 481-1023 tablet, >=1024 desktop grid)
    - _Requirements: 14.1, 14.2, 14.3_

  - [ ] 16.2 Implement the API client and shared components
    - Add a typed API client (HTTPS/JSON) for all endpoints and shared components `Card`, `LoadingSkeleton`, `UnavailableState`, `ErrorBoundary`, `FocusableControl` with a visible keyboard focus indicator and contrast-compliant color tokens
    - _Requirements: 8.4, 8.5, 14.4, 19.1, 19.3_

  - [ ]* 16.3 Write component tests for shared components
    - Cover `ErrorBoundary` containment, `UnavailableState` rendering, `LoadingSkeleton` while pending, and focus-indicator presence
    - _Requirements: 8.4, 8.5, 19.3_

- [ ] 17. Frontend feature cards and charts
  - [ ] 17.1 Implement DiscountCheckerCard with the SHAP waterfall
    - Show the classification label (text + color) as the primary conclusion; expand on hover/keyboard to a Plotly SHAP waterfall with base value and contributions
    - _Requirements: 2.4, 3.2, 8.3, 19.2, 19.4_

  - [ ] 17.2 Implement ShrinkflationCard with the timeline chart
    - Primary conclusion first; expandable Recharts timeline of pack size and unit price with source attribution and a text alternative
    - _Requirements: 4.1, 4.4, 8.3, 19.4, 19.5_

  - [ ] 17.3 Implement UnitPriceCard with comparison bars
    - Show the best-value variant first; expand to per-variant price/quantity/unit-price bars with a text alternative
    - _Requirements: 5.2, 5.3, 8.3, 19.4, 19.5_

  - [ ] 17.4 Implement BuyTimingCard with the seasonality view
    - Show buy-now/wait first with the category-level + snapshot disclosure; expand to the seasonal window view
    - _Requirements: 6.1, 6.2, 6.4, 8.3, 10.1_

  - [ ] 17.5 Implement CrossPlatformCard with platform bars
    - Show the best deal first; expand to per-platform prices with product links and any genuineness score
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.3_

  - [ ]* 17.6 Write fast-check property test and component tests for the cards
    - **Property 25: Non-color and text-alternative cues are always present** - _Validates: Requirements 19.2, 19.5_
    - Also cover hover/keyboard expand and conclusion-before-detail ordering
    - _Requirements: 8.3, 19.4_

- [ ] 18. Frontend pages and disclosure panels
  - [ ] 18.1 Implement SearchPage
    - Build the query box, results list (name/brand/category), empty-query prompt, no-results message, and manual-entry form wired to the search/manual-entry endpoints
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6_

  - [ ] 18.2 Implement DashboardPage with the compact responsive grid
    - Compose `ProductHeader`, `DisclosureBanner`, the five-card `FeatureGrid`, and `DataSourcesPanel`; wire per-card `ErrorBoundary` and `LoadingSkeleton`; ensure each module's primary result fits one screen at >=1024px and the grid reflows to tablet/single-column
    - _Requirements: 8.1, 8.2, 8.4, 8.5, 10.1, 10.2, 14.2, 14.3_

  - [ ]* 18.3 Write responsive and accessibility tests for the pages
    - Snapshot at 480/800/1280 px; `jest-axe` checks for contrast tokens, text-plus-color labels, and chart text alternatives
    - _Requirements: 8.2, 14.1, 14.2, 14.3, 19.1, 19.2, 19.5_

- [ ] 19. Testing consolidation and quality gates
  - [ ]* 19.1 Verify full property-test coverage of all 25 correctness properties
    - Confirm each of Properties 1-25 has exactly one property-based test (Hypothesis backend / fast-check frontend) with the traceability tag `# Feature: price-truth-platform, Property {n}` and `max_examples`/`numRuns` >= 100
    - _Requirements: 17.1_

  - [ ]* 19.2 Add API contract and latency/load tests
    - `TestClient` contract tests (JSON + status + error schema) for every endpoint; Locust/k6 checks for cache-hit p95 <=500 ms at 50 concurrency, error rate <=1%, discount <3 s and SHAP <2 s warm
    - _Requirements: 11.1, 11.2, 11.3, 12.1, 14.4, 15.3_

  - [ ]* 19.3 Configure coverage gate and linting in CI
    - Add `pytest --cov` with a >=70% gate on backend business-logic modules; wire `black`/`isort`/`flake8`/`mypy` for Python and `eslint`/`prettier` for JavaScript to run zero-error in a `.github/workflows/` CI file
    - _Requirements: 17.2, 17.3, 17.4_

- [ ] 20. Deployment configuration and smoke test
  - [ ] 20.1 Finalize container and environment configuration
    - Ensure `docker-compose.yml` brings up backend + Postgres + Redis with one command, `.env.example` documents all env vars, and `.env`/secrets are gitignored and excluded from responses
    - _Requirements: 13.1, 13.2, 13.4, 18.6_

  - [ ] 20.2 Configure Vercel (frontend) and Railway (backend/DB/Redis) deploys
    - Add Vercel config for the static frontend build (HTTPS/CDN) and Railway config for the backend, Postgres, and Redis on free tiers, reading all configuration from environment variables
    - _Requirements: 13.3, 18.5_

  - [ ]* 20.3 Add a deployed smoke test
    - Script a post-deploy check that `/health` returns healthy, `/docs` and `/redoc` return 200, and a sample discount-check and unit-price compare return valid JSON
    - _Requirements: 16.1, 16.3, 17.4_

- [ ] 21. Final checkpoint - full stack
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks (property, unit, integration, contract, accessibility) and can be skipped for a faster MVP; core implementation tasks are never optional.
- Every task references specific requirement clauses, and every property test references its Correctness Property number for full traceability.
- The synthetic fixtures (Task 3.1) keep the entire pipeline, ML training, and all feature services runnable and testable before the real Kaggle CSVs are downloaded; only Task 3.5 depends on the user obtaining Kaggle credentials and data.
- Checkpoints (Tasks 5, 15, 21) provide incremental validation points; each numbered sub-task is small enough to commit individually to support continuous incremental git commits.
- Feature services are ordered to build the most property-testable component (Unit Price Comparator) first, and the Data Service is sequenced before its search/shrinkflation consumers.

## Task Dependency Graph

Waves are ordered so that no two tasks in the same wave write to the same file. In particular, every task that edits `main.py` (router registration, middleware, startup model load: 1.2, 1.5, 6.3, 9.3, 10.3, 4.4, 11.3, 12.3, 8.3, 13.1, 14.1, 14.2) is isolated to its own wave, as are the sequential edits to `data/scripts/ingest.py` (3.2, 3.3, 3.4, 3.5), `app/ml/discount_model.py` (4.1, 4.2, 4.4), `app/services/data_service.py` (7.1, 7.2, 7.3), and `app/api/v1/meta.py` (1.5, 13.2).

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "6.1"] },
    { "id": 2, "tasks": ["1.3", "2.1", "6.2"] },
    { "id": 3, "tasks": ["1.5", "2.2", "7.1"] },
    { "id": 4, "tasks": ["1.6", "2.3", "3.2", "7.2", "9.1", "10.1", "11.1", "12.1"] },
    { "id": 5, "tasks": ["3.3", "4.1", "6.3", "7.3", "9.2", "10.2", "11.2", "12.2"] },
    { "id": 6, "tasks": ["3.4", "3.6", "4.2", "6.4", "7.4", "7.5", "9.3"] },
    { "id": 7, "tasks": ["3.5", "4.3", "9.4", "10.3"] },
    { "id": 8, "tasks": ["4.4", "10.4"] },
    { "id": 9, "tasks": ["4.5", "8.1", "11.3"] },
    { "id": 10, "tasks": ["4.6", "8.2", "11.4", "12.3"] },
    { "id": 11, "tasks": ["8.3", "12.4"] },
    { "id": 12, "tasks": ["8.4", "8.5", "13.1", "13.2"] },
    { "id": 13, "tasks": ["13.3", "14.1", "16.1"] },
    { "id": 14, "tasks": ["14.2", "16.2"] },
    { "id": 15, "tasks": ["14.3", "16.3", "17.1", "17.2", "17.3", "17.4", "17.5", "18.1", "20.1"] },
    { "id": 16, "tasks": ["17.6", "18.2", "19.2"] },
    { "id": 17, "tasks": ["18.3", "19.1", "20.2"] },
    { "id": 18, "tasks": ["19.3", "20.3"] }
  ]
}
```
