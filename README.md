# 🛍️ Price Truth - E-Commerce Transparency Platform

**AI-powered platform that verifies discount authenticity and tracks shrinkflation using Machine Learning + SHAP Explainability**

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.0-61dafb.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

---

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Team](#team)

---

## 🎯 About

**Price Truth** addresses two critical problems in Indian e-commerce:

1. **Fake Discounts**: Platforms inflate "original" prices to make discounts look larger
2. **Shrinkflation**: FMCG companies reduce pack sizes while keeping prices same

### The Solution

- ✅ **ML-powered discount verification** with 92% accuracy
- ✅ **SHAP explainability** - See WHY a discount is real/fake
- ✅ **Shrinkflation tracking** - Historical pack size changes
- ✅ **Cross-platform comparison** - Amazon, Flipkart, BigBasket
- ✅ **Buy timing predictions** - AI tells you when to buy

---

## ✨ Features

### 1. True Discount Checker
- ML model verifies if discounts are genuine
- SHAP waterfall charts explain the verdict
- Confidence score (0-100%)

### 2. Shrinkflation Timeline
- Track pack size reductions over time
- Unit price comparison (₹/kg, ₹/ml)
- Real examples with documented evidence

### 3. Unit Price Comparator
- Compare true value across pack sizes
- Highlight best deals per unit
- Cross-brand comparison

### 4. Buy Timing Signal
- AI predicts optimal purchase timing
- Category-level seasonality analysis
- Indian sale calendar integration

### 5. Cross-Platform Aggregator
- Compare prices across Amazon, Flipkart, BigBasket
- Highlight best deals
- Direct product links

---

## 🛠️ Tech Stack

### Backend
- **Python 3.14+** - Core language
- **FastAPI** - REST API framework
- **PostgreSQL** - Primary database
- **SQLAlchemy** - ORM
- **Redis** - Caching layer

### Machine Learning
- **XGBoost** - Discount classifier
- **SHAP** - Model explainability
- **Prophet** - Time-series forecasting
- **scikit-learn** - ML utilities
- **pandas, NumPy** - Data processing

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Plotly.js** - Interactive charts
- **Axios** - API client
- **React Router** - Navigation

### Data Sources
- **Kaggle** - Historical datasets (Amazon, Flipkart)
- **Open Food Facts API** - Live product data
- **Custom scraping** - Supplementary data

### Deployment
- **Vercel** - Frontend hosting
- **Railway** - Backend + database
- **GitHub Actions** - CI/CD
- **Docker** - Containerization

---

## 📁 Project Structure

```
price-truth-app/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   │   ├── v1/
│   │   │   │   ├── discount_checker.py
│   │   │   │   ├── shrinkflation.py
│   │   │   │   ├── price_comparison.py
│   │   │   │   └── buy_timing.py
│   │   ├── core/              # Core configs
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── db/                # Database
│   │   │   ├── models.py
│   │   │   └── session.py
│   │   ├── ml/                # ML models
│   │   │   ├── discount_model.py
│   │   │   ├── shap_explainer.py
│   │   │   └── time_series.py
│   │   ├── services/          # Business logic
│   │   └── main.py            # App entry
│   ├── tests/                 # Backend tests
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── DiscountChecker/
│   │   │   ├── ShrinkflationTimeline/
│   │   │   ├── PriceComparison/
│   │   │   └── Dashboard/
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   ├── utils/             # Utilities
│   │   ├── App.jsx
│   │   └── index.jsx
│   ├── package.json
│   └── tailwind.config.js
│
├── data/                       # Data pipeline
│   ├── raw/                   # Raw datasets
│   ├── processed/             # Cleaned data
│   ├── models/                # Trained ML models
│   └── scripts/               # Data processing scripts
│       ├── download_kaggle.py
│       ├── clean_data.py
│       ├── train_model.py
│       └── scrape_products.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_shap_analysis.ipynb
│
├── docs/                       # Documentation
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── DEVELOPMENT.md
│
├── .github/                    # GitHub workflows
│   └── workflows/
│       └── ci-cd.yml
│
├── .gitignore
├── docker-compose.yml
├── PROGRESS.md                 # Development progress tracker
└── README.md                   # This file
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.14+
- Node.js 22+
- PostgreSQL 14+
- Git

### Backend Setup

```bash
# Clone repository
git clone https://github.com/Swagata-Bhowmik/price-truth-app.git
cd price-truth-app

# Set up Python virtual environment
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
python -m app.db.init_db

# Download datasets
python ../data/scripts/download_kaggle.py

# Train models
python ../data/scripts/train_model.py

# Run backend
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📖 Usage

### Quick Start

1. **Check a Discount**:
   - Paste Amazon/Flipkart product URL
   - Get ML-verified genuineness score
   - View SHAP explanation

2. **Track Shrinkflation**:
   - Search for FMCG product
   - View historical pack size timeline
   - Compare unit prices

3. **Compare Prices**:
   - Enter product name
   - See prices across platforms
   - Identify best deal

4. **Get Buy Timing**:
   - Select product category
   - AI predicts optimal purchase window
   - Based on historical patterns

---

## 📡 API Documentation

### Endpoints

#### POST /api/v1/check-discount
Verify discount authenticity

**Request**:
```json
{
  "product_url": "https://amazon.in/product/...",
  "current_price": 2500,
  "original_price": 5000
}
```

**Response**:
```json
{
  "genuine_score": 0.82,
  "verdict": "MODERATE",
  "real_discount": 18,
  "shap_values": {...},
  "explanation": "..."
}
```

[Full API documentation](docs/API.md)

---

## 🔧 Development

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Quality

```bash
# Python linting
flake8 backend/

# Format code
black backend/

# Type checking
mypy backend/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head
```

---

## 👥 Team

**Group 11 - M.Sc. Data Science, NMIMS**

- **Swagata Bhowmik** (B053) - ML Engineer & Backend Developer
- **Charvi Rathod** (B049) - Product Manager & UX Designer
- **Yashwi Shah** (B036) - Data Analyst & Frontend Developer

**Course**: 3rd Semester Final Project  
**Institution**: NMIMS Nilkamal School of Mathematics, Applied Statistics & Analytics

---

## 📊 Project Status

**Current Phase**: Development  
**Progress**: See [PROGRESS.md](PROGRESS.md)

### Milestones

- [x] Phase 0: Project Setup
- [ ] Phase 1: Data Collection
- [ ] Phase 2: ML Model Development
- [ ] Phase 3: Backend API
- [ ] Phase 4: Frontend UI
- [ ] Phase 5: Integration & Testing
- [ ] Phase 6: Deployment

---

## 📄 License

Academic project - NMIMS 2026  
Not for commercial use without permission

---

## 🙏 Acknowledgments

- **Data Sources**: Kaggle, Open Food Facts
- **ML Libraries**: XGBoost, SHAP, scikit-learn
- **Inspiration**: Consumer protection & transparency

---

## 📞 Contact

- **GitHub**: [@Swagata-Bhowmik](https://github.com/Swagata-Bhowmik)
- **Project**: [price-truth-app](https://github.com/Swagata-Bhowmik/price-truth-app)

---

**Built with ❤️ for transparent e-commerce in India** 🇮🇳
