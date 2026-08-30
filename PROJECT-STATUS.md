# PRICE TRUTH - PROJECT STATUS & HANDOFF

**Last updated:** August 30, 2026
**Purpose:** Single source of truth for current state. Read this first before continuing development. Created after a model switch to preserve full context.

---

## 1. WHAT THIS PROJECT IS

Price Truth - an AI-powered web platform bringing transparency to Indian e-commerce pricing.

Two core problems it solves:
1. Fake discounts - inflated "original" prices that make discounts look bigger than they are.
2. Shrinkflation - pack sizes shrinking over time while prices stay the same.

Five finalized features (locked from the presentation):
| # | Feature | What it does |
|---|---------|--------------|
| 1 | True Discount Checker | ML + SHAP verifies whether a discount is genuine, with an explainable confidence score |
| 2 | Shrinkflation Timeline | Tracks pack-size reduction over time and the hidden unit-price increase |
| 3 | Unit Price Comparator | Compares true value (per g, per ml) across pack sizes and brands |
| 4 | Buy Timing Signal | Predicts the best time to buy using price-trend analysis |
| 5 | Cross-Platform Aggregator | Compares prices across Amazon, Flipkart, Croma, Tata CLiQ, Reliance Digital |

Team: Swagata Bhowmik (B053), Charvi Rathod (B049), Yashwi Shah (B036)
Course: M.Sc. Data Science, 3rd Sem - NMIMS
GitHub: https://github.com/Swagata-Bhowmik

---

## 2. KEY DECISIONS ALREADY MADE (do not re-litigate)

| Decision | Choice | Why |
|----------|--------|-----|
| Budget | Zero - free tools only | No money to be spent |
| Manual data collection | None | User has no time; everything automated |
| Data sources | Kaggle datasets + Open Food Facts API + light automated scraping | All free and legal |
| Scope target | High-quality working product (not just a demo) | Goes into GitHub as a portfolio piece |
| Backend | Python + FastAPI | Async, auto-docs, ML-friendly |
| Frontend | React 18 + Tailwind CSS | Interactive, modern, compact dashboard |
| ML | XGBoost + SHAP + Prophet | Explainable, tabular-friendly |
| Database | PostgreSQL + Redis | Reliable + caching |
| Deployment | Vercel (frontend) + Railway (backend/db) | Free tiers |
| Git strategy | Continuous incremental commits | Show progress over time, not one big push |

Dashboard vision (user explicit ask): lively, compact (fits one screen ratio), interactive hover-to-expand, clean, strong color contrast.

---

## 3. CURRENT STATE - WHAT EXISTS RIGHT NOW

### Phase 0 COMPLETE - Project scaffolding

Location: c:\Users\Lenovo\Downloads\price-truth-app\

Git: local repo initialized, branch master, 2 commits
- 29cd212  Initial commit: project setup
- 1e0b9bc  Phase 0 complete: progress tracker

Folder structure created:
  backend/
    app/
      api/v1/     (empty module)
      core/       (empty module)
      db/         (empty module)
      ml/         (empty module)
      services/   (empty module)
      main.py     - FastAPI skeleton (root + /health endpoints)
    tests/        (empty)
    requirements.txt - all Python deps listed
  frontend/
    package.json  - React 18 + Tailwind deps listed (NOT installed yet)
  data/
    raw/ processed/ models/  (empty)
    scripts/download_kaggle_datasets.py - written, UNCOMMITTED, untested
  notebooks/  docs/  .github/workflows/  (empty)
  .gitignore, docker-compose.yml, README.md, PROGRESS.md

### Important truths about current state
- No code actually runs yet beyond the FastAPI hello-world skeleton.
- No dependencies installed (no venv created, no npm install run).
- No datasets downloaded.
- No ML model exists.
- No GitHub remote connected - commits are local only.
- download_kaggle_datasets.py is written but untested and uncommitted.

---

## 4. WHAT IS NOT DONE (the real work ahead)

Phases 1-12 from PROGRESS.md are all pending:
1. Data collection and preparation
2. Data cleaning and feature engineering
3. ML model - discount detector
4. SHAP explainability
5. Time-series - buy timing
6. Backend API
7. Frontend React app
8. Shrinkflation feature
9. Cross-platform aggregator
10. Testing and QA
11. Deployment
12. Documentation and demo

---

## 5. OPEN ITEMS NEEDING ATTENTION

- [ ] Connect GitHub remote and push the 2 local commits
- [ ] Decide the data approach in detail (which exact Kaggle datasets; how each feature gets real data)
- [ ] Resolve the data honesty gap: public datasets are price snapshots, not daily time series -> Buy Timing must be category-level, not per-product-per-day
- [ ] Confirm shrinkflation data strategy (Open Food Facts history vs curated public examples)
- [ ] Verify Kaggle API credentials are available on this machine

---

## 6. ENVIRONMENT (verified)

- Python: 3.14.3
- Node.js: v22.22.2
- Git: configured as "Swagata Bhowmik"
- OS: Windows / PowerShell

---

## 7. NEXT STEP

Before writing more code, we are creating a formal development spec (requirements -> design -> tasks) so the build is planned bit-by-bit, verified at each step, and committed incrementally. This document will be updated as each phase closes.

---

## BUILD ENVIRONMENT NOTE (Task 1.1 - critical)

- The backend virtualenv is **Python 3.11.9**, NOT the machine default 3.14.3. Reason: the pinned ML stack (numpy 1.26, pandas, scikit-learn, xgboost, shap, prophet, statsmodels) has no Python 3.14 wheels.
- venv location: `backend\venv`  ->  use `backend\venv\Scripts\python.exe` (or activate the venv) for ALL backend commands, tests, and scripts. Bare `python` (3.14) will fail to import the ML libs.
- All backend dependencies installed successfully at their pinned versions.
- Config: `backend/app/core/config.py` (pydantic-settings) reads DATABASE_URL, REDIS_URL, OFF_BASE_URL, OFF_VERSION, CORS_ALLOWED_ORIGIN from env; template in `backend/.env.example`.
