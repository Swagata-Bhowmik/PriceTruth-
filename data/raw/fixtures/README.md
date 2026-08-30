# Synthetic Fixture Datasets

**These files are SYNTHETIC data generated for development and testing only.**
They are **not** real scraped data and were **not** downloaded from Kaggle,
Amazon, Flipkart, or any live site. Every product, price, rating, and source
citation here is fabricated. Do not present any value in these files as a real
market price or a real published statistic.

## Why these exist

Task 3.1 of the `price-truth-platform` spec. These fixtures let the entire
downstream pipeline run and be tested **without** the real Kaggle download:

- ingestion / cleaning (task 3.2)
- category price statistics (task 3.3)
- category seasonality (task 3.4)
- XGBoost discount model training + SHAP (task 4)
- every feature service (discount, shrinkflation, unit price, buy timing,
  cross-platform)

Only task 3.5 (the full real-data run) needs the actual Kaggle CSVs; everything
before it can use these fixtures.

## Files and columns

### `amazon_sample.csv` (78 rows)
Mirrors the Kaggle **Amazon Sales Dataset** (`karkavelrajaj/amazon-sales-dataset`).

| Column | Notes |
| --- | --- |
| `product_id` | `amz_XXXX`. Anchors `amz_0001`..`amz_0007`. |
| `product_name` | Brand + item + a short model code. |
| `category` | Amazon-style pipe hierarchy, e.g. `Electronics|Audio|Headphones`. |
| `discounted_price` | Selling price. **Mostly plain numbers**; a few rows use `₹`/comma formatting (e.g. `₹1,299`). |
| `actual_price` | Reference / "original" price. Same mixed formatting. |
| `discount_percentage` | Integer percent; a few rows are suffixed with `%`. |
| `rating` | 0-5. |
| `rating_count` | Integer; a few rows use comma grouping (e.g. `32,210`). |

6 rows deliberately carry `₹` symbols, thousands separators, and `%` signs
(covering both a genuine and an inflated row) so the ingestion **cleaning logic
has something to strip and can be unit-tested** (task 3.2 / Req 9.5).

### `flipkart_sample.csv` (54 rows)
Mirrors the Kaggle **Flipkart e-commerce Dataset**
(`atharvjairath/flipkart-ecommerce-dataset`).

| Column | Notes |
| --- | --- |
| `product_id` | `fk_XXXX`. Anchor `fk_0001`. |
| `product_name` | Brand + item + model code. |
| `category_tree` | Flipkart-style bracketed tree, e.g. `["Electronics >> Audio >> Headphones"]`. |
| `retail_price` | Reference / "original" price. |
| `discounted_price` | Selling price. |
| `brand` | Brand name. |
| `overall_rating` | 0-5. |
| `rating_count` | Integer. |

Flipkart values are kept as plain numbers (the real Flipkart export is numeric);
the "messy" currency formatting lives only in `amazon_sample.csv`.

### `pack_size_history.csv` (11 rows)
Curated shrinkflation records feeding `pack_size_history`
(Shrinkflation Timeline, Req 4).

| Column | Notes |
| --- | --- |
| `product_id` | Joins back to an anchor in `amazon_sample.csv`. |
| `product_name` | |
| `observed_at` | `YYYY-MM-DD`. |
| `pack_quantity` | Numeric quantity. |
| `pack_unit` | `g` or `ml`. |
| `selling_price` | Held roughly constant across a series to illustrate shrinkflation. |
| `source_type` | `off` (Open Food Facts quantity field) or `cited_public_record`. |
| `source_citation` | Plausible citation string, each tagged `(synthetic fixture)`. |

Series included:
- **Parle-G** (`amz_0001`): 4 points, 2019-2025, 100 g → 75 g at a steady ₹10.
- **Dove shampoo** (`amz_0006`): 3 points, 400 ml → 340 ml at a steady ₹299.
- **Colgate toothpaste** (`amz_0007`): 3 points, 150 g → 132 g at a steady ₹99.
- **Fortune oil** (`amz_0002`): 1 point (exercises the "history exists but
  fewer than two points, so no percentage change" branch, Req 4.3).

### `platform_prices.csv` (18 rows)
Same product sold across Supported Platforms (Cross-Platform Aggregator, Req 7).

| Column | Notes |
| --- | --- |
| `product_id` | Joins to anchors in the amazon/flipkart samples. |
| `product_name` | |
| `platform` | One of `Amazon`, `Flipkart`, `Croma`, `Tata CLiQ`, `Reliance Digital`. |
| `price` | Platform price. |
| `product_url` | Plausible product-page link. |

Coverage:
- `amz_0003` (boAt neckband): all **5** platforms → best deal Flipkart ₹1249.
- `amz_0004` (Samsung TV): **4** platforms → best deal Flipkart ₹31499.
- `amz_0005` (Prestige cooker): **3** platforms.
- `fk_0001` (JBL headphones): **3** platforms (demonstrates a Flipkart-origin id join).
- `amz_0001` (Parle-G): **2** platforms (grocery lives only on Amazon/Flipkart).
- `amz_0006` (Dove shampoo): **1** platform → exercises the no-comparison path (Req 7.5).

## Categories

Six canonical categories, ~22 product rows each (14 genuine + 8 inflated),
132 rows total across the amazon + flipkart samples, so per-category
mean/median/std price statistics are stable (task 3.3).

| Canonical slug | Amazon `category` | Flipkart `category_tree` leaf |
| --- | --- | --- |
| `electronics/headphones` | `Electronics\|Audio\|Headphones` | `... >> Audio >> Headphones` |
| `electronics/tv` | `Electronics\|Televisions\|SmartTVs` | `... >> Televisions >> Smart TVs` |
| `grocery/biscuits` | `Grocery&GourmetFoods\|Biscuits` | `... >> Snacks >> Biscuits` |
| `grocery/edible-oil` | `Grocery&GourmetFoods\|EdibleOil` | `... >> Cooking Essentials >> Edible Oil` |
| `home/kitchen` | `Home&Kitchen\|Kitchen` | `... >> Cookware >> Kitchen Tools` |
| `personal-care` | `Beauty&PersonalCare\|PersonalCare` | `... >> Personal Care` |

The ingestion step (task 3.2) normalizes both the Amazon pipe hierarchy and the
Flipkart bracketed tree onto these canonical slugs (keyword match on the leaf).

## Genuine vs inflated discount patterns

Both discount patterns are present in every category so the weak-supervision
labeling (task 4.2) has clean signal:

- **Genuine** rows: the discounted price sits genuinely **below** the category
  norm, with a reference price in the normal band (~13-33% off).
- **Inflated** rows: the discounted price sits **near** the category norm while
  the reference/`actual` price is a high outlier (~64-79% "off") — a
  manufactured discount built by inflating the "original" price.

## Reproducing / regenerating

`generate_fixtures.py` (stdlib-only, fixed RNG seed `20240517`) produces all
four CSVs deterministically:

```powershell
backend\venv\Scripts\python.exe data\raw\fixtures\generate_fixtures.py
```

Re-running overwrites the CSVs with byte-identical content. The curated
`pack_size_history` and `platform_prices` rows are hard-coded in the generator
(they are meant to be stable, cited records rather than random draws).
