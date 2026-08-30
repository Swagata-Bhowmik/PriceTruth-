"""
generate_fixtures.py - Deterministic synthetic fixture generator for Price Truth.

Produces SYNTHETIC (not real / not scraped) CSV datasets under
``data/raw/fixtures/`` that mirror the Kaggle "Amazon Sales" and
"Flipkart e-commerce" schemas, plus curated shrinkflation
(``pack_size_history``) and cross-platform (``platform_prices``) records.

Goal: make the entire downstream pipeline - ingestion (task 3.2), category
price statistics (3.3), seasonality (3.4), ML training (task 4) and every
feature service - runnable and testable WITHOUT the real Kaggle download.

Design references (`.kiro/specs/price-truth-platform/design.md`):
  * Data Models section -> tables products, category_price_stats,
    price_snapshots, pack_size_history, platform_prices, category_seasonality.
  * Discount model weak-supervision rule (4.2): a row is "inflated" when the
    reference/actual price is a category-distribution outlier ABOVE the norm
    while the discounted price sits NEAR the norm; "genuine" when the
    discounted price is genuinely BELOW the norm with a reference in the
    normal band. This generator deliberately produces both clusters so the
    later labeling has clean signal.

Usage:
    backend\\venv\\Scripts\\python.exe data/raw/fixtures/generate_fixtures.py

The script is stdlib-only (csv, random, pathlib) so it runs on any Python,
and uses a fixed RNG seed so output is byte-stable across runs.
"""

import csv
import random
from pathlib import Path

SEED = 20240517
OUT_DIR = Path(__file__).resolve().parent

# Supported Platforms per the requirements glossary.
PLATFORMS = ["Amazon", "Flipkart", "Croma", "Tata CLiQ", "Reliance Digital"]

# ---------------------------------------------------------------------------
# Per-category configuration.
#   slug         : canonical category used across the platform / joins
#   amazon_cat   : realistic Amazon-style pipe hierarchy (leaf keyword clear)
#   flipkart_tree: realistic Flipkart-style bracketed category tree
#   T            : typical selling (discounted) price center in INR
#   rating range / rating_count range : realistic bounds
#   brands / names : building blocks for plausible product names
# ---------------------------------------------------------------------------
CATEGORIES = [
    {
        "slug": "electronics/headphones",
        "amazon_cat": "Electronics|Audio|Headphones",
        "flipkart_tree": '["Electronics >> Audio >> Headphones"]',
        "T": 1600, "r_lo": 3.6, "r_hi": 4.6, "rc_lo": 150, "rc_hi": 40000,
        "brands": ["boAt", "JBL", "Sony", "Noise", "OnePlus", "Realme", "pTron"],
        "names": ["Wireless Headphones", "Bluetooth Earbuds", "Over-Ear Headphones",
                  "Neckband Earphones", "Gaming Headset", "True Wireless Earbuds"],
    },
    {
        "slug": "electronics/tv",
        "amazon_cat": "Electronics|Televisions|SmartTVs",
        "flipkart_tree": '["Electronics >> Televisions >> Smart TVs"]',
        "T": 34000, "r_lo": 3.9, "r_hi": 4.7, "rc_lo": 60, "rc_hi": 12000,
        "brands": ["Samsung", "LG", "Sony", "Mi", "OnePlus", "TCL", "Acer"],
        "names": ["32 inch HD Ready LED TV", "43 inch Full HD Smart TV",
                  "50 inch 4K UHD Smart TV", "55 inch QLED Smart TV", "40 inch Android TV"],
    },
    {
        "slug": "grocery/biscuits",
        "amazon_cat": "Grocery&GourmetFoods|Biscuits",
        "flipkart_tree": '["Food & Beverages >> Snacks >> Biscuits"]',
        "T": 40, "r_lo": 4.0, "r_hi": 4.8, "rc_lo": 400, "rc_hi": 95000,
        "brands": ["Parle", "Britannia", "Sunfeast", "Unibic", "McVitie's", "Cadbury"],
        "names": ["Glucose Biscuits", "Cream Biscuits", "Digestive Biscuits",
                  "Marie Light Biscuits", "Choco Chip Cookies", "Bourbon Biscuit Pack"],
    },
    {
        "slug": "grocery/edible-oil",
        "amazon_cat": "Grocery&GourmetFoods|EdibleOil",
        "flipkart_tree": '["Food & Beverages >> Cooking Essentials >> Edible Oil"]',
        "T": 175, "r_lo": 3.8, "r_hi": 4.6, "rc_lo": 250, "rc_hi": 42000,
        "brands": ["Fortune", "Saffola", "Dhara", "Sundrop", "Gemini", "Figaro"],
        "names": ["Refined Sunflower Oil 1L", "Kachi Ghani Mustard Oil 1L",
                  "Groundnut Oil 1L", "Soybean Oil 1L", "Rice Bran Oil 1L"],
    },
    {
        "slug": "home/kitchen",
        "amazon_cat": "Home&Kitchen|Kitchen",
        "flipkart_tree": '["Home & Kitchen >> Cookware >> Kitchen Tools"]',
        "T": 900, "r_lo": 3.6, "r_hi": 4.6, "rc_lo": 80, "rc_hi": 26000,
        "brands": ["Prestige", "Pigeon", "Milton", "Cello", "Butterfly", "Hawkins"],
        "names": ["3L Pressure Cooker", "Non-Stick Kadai", "Stainless Steel Casserole Set",
                  "1L Vacuum Flask", "Hand Blender 300W", "Vegetable Chopper 900ml"],
    },
    {
        "slug": "personal-care",
        "amazon_cat": "Beauty&PersonalCare|PersonalCare",
        "flipkart_tree": '["Beauty & Personal Care >> Personal Care"]',
        "T": 210, "r_lo": 3.9, "r_hi": 4.7, "rc_lo": 300, "rc_hi": 62000,
        "brands": ["Dove", "Himalaya", "Nivea", "Mamaearth", "L'Oreal", "Garnier"],
        "names": ["Daily Shampoo", "Gentle Face Wash", "Body Lotion",
                  "Nourishing Hair Oil", "Moisturising Cream", "Sunscreen SPF 50"],
    },
]

# ---------------------------------------------------------------------------
# Curated "anchor" products with FIXED ids so pack_size_history and
# platform_prices can reference them (cross-file joins for demos/tests).
# All anchors use a genuine (normal-reference) discount pattern.
#   (id, name, slug, file, discounted, actual, brand, rating, rating_count)
# ---------------------------------------------------------------------------
ANCHORS = [
    ("amz_0001", "Parle-G Original Glucose Biscuits", "grocery/biscuits",
     "amazon", 10.0, 12.0, "Parle", 4.4, 84210),
    ("amz_0002", "Fortune Sunlite Refined Sunflower Oil", "grocery/edible-oil",
     "amazon", 155.0, 199.0, "Fortune", 4.3, 38940),
    ("amz_0003", "boAt Rockerz 255 Pro+ Bluetooth Neckband", "electronics/headphones",
     "amazon", 1299.0, 1999.0, "boAt", 4.1, 32210),
    ("amz_0004", "Samsung 43 inch Crystal 4K UHD Smart TV", "electronics/tv",
     "amazon", 32999.0, 47900.0, "Samsung", 4.3, 11200),
    ("amz_0005", "Prestige Deluxe Alpha Stainless Steel Pressure Cooker", "home/kitchen",
     "amazon", 1199.0, 1650.0, "Prestige", 4.4, 20130),
    ("amz_0006", "Dove Intense Repair Daily Shampoo", "personal-care",
     "amazon", 299.0, 399.0, "Dove", 4.4, 51220),
    ("amz_0007", "Colgate MaxFresh Red Gel Toothpaste", "personal-care",
     "amazon", 99.0, 120.0, "Colgate", 4.5, 60110),
    ("fk_0001", "JBL Tune 510BT Wireless On-Ear Headphones", "electronics/headphones",
     "flipkart", 2499.0, 3999.0, "JBL", 4.2, 15420),
]

# Curated shrinkflation records (pack shrinks while price held ~constant).
# source_type is one of {off, cited_public_record}. Citations are explicitly
# labelled "(synthetic fixture)" because this is generated demo data.
PACK_SIZE_HISTORY = [
    # Parle-G: 4-point series 2019 -> 2025, grams down, price steady at Rs.10
    ("amz_0001", "Parle-G Original Glucose Biscuits", "2019-03-01", 100, "g", 10.0,
     "cited_public_record", "Consumer press coverage of Parle-G Rs.10 pack grammage, 2019 (synthetic fixture)"),
    ("amz_0001", "Parle-G Original Glucose Biscuits", "2021-06-01", 92, "g", 10.0,
     "cited_public_record", "Retail shelf audit archive, 2021 (synthetic fixture)"),
    ("amz_0001", "Parle-G Original Glucose Biscuits", "2023-05-01", 83, "g", 10.0,
     "cited_public_record", "Shrinkflation tracker community dataset, 2023 (synthetic fixture)"),
    ("amz_0001", "Parle-G Original Glucose Biscuits", "2025-02-01", 75, "g", 10.0,
     "off", "Open Food Facts product quantity field, 2025 (synthetic fixture)"),
    # Dove shampoo: 3-point series, millilitres down, price steady at Rs.299
    ("amz_0006", "Dove Intense Repair Daily Shampoo", "2020-01-01", 400, "ml", 299.0,
     "cited_public_record", "Archived e-commerce listing snapshot, 2020 (synthetic fixture)"),
    ("amz_0006", "Dove Intense Repair Daily Shampoo", "2022-07-01", 380, "ml", 299.0,
     "cited_public_record", "Consumer forum price/size log, 2022 (synthetic fixture)"),
    ("amz_0006", "Dove Intense Repair Daily Shampoo", "2024-08-01", 340, "ml", 299.0,
     "off", "Open Food Facts product quantity field, 2024 (synthetic fixture)"),
    # Colgate toothpaste: 3-point series, grams down, price steady at Rs.99
    ("amz_0007", "Colgate MaxFresh Red Gel Toothpaste", "2021-01-01", 150, "g", 99.0,
     "cited_public_record", "Packaging archive comparison, 2021 (synthetic fixture)"),
    ("amz_0007", "Colgate MaxFresh Red Gel Toothpaste", "2023-03-01", 140, "g", 99.0,
     "cited_public_record", "Retail audit newsletter, 2023 (synthetic fixture)"),
    ("amz_0007", "Colgate MaxFresh Red Gel Toothpaste", "2025-01-01", 132, "g", 99.0,
     "off", "Open Food Facts product quantity field, 2025 (synthetic fixture)"),
    # Fortune oil: single point (history exists but <2 points -> no %-change branch)
    ("amz_0002", "Fortune Sunlite Refined Sunflower Oil", "2024-01-01", 1000, "ml", 155.0,
     "off", "Open Food Facts product quantity field, 2024 (synthetic fixture)"),
]

# Curated cross-platform price rows (product on multiple Supported Platforms).
#   (product_id, product_name, platform, price, product_url)
PLATFORM_PRICES = [
    # boAt neckband on all 5 platforms -> best deal = Flipkart 1249
    ("amz_0003", "boAt Rockerz 255 Pro+ Bluetooth Neckband", "Amazon", 1299.0,
     "https://www.amazon.in/dp/B0BOAT255P"),
    ("amz_0003", "boAt Rockerz 255 Pro+ Bluetooth Neckband", "Flipkart", 1249.0,
     "https://www.flipkart.com/boat-rockerz-255-pro/p/itmboat255pro"),
    ("amz_0003", "boAt Rockerz 255 Pro+ Bluetooth Neckband", "Croma", 1349.0,
     "https://www.croma.com/boat-rockerz-255-pro/p/244501"),
    ("amz_0003", "boAt Rockerz 255 Pro+ Bluetooth Neckband", "Tata CLiQ", 1279.0,
     "https://www.tatacliq.com/boat-rockerz-255-pro/p-mp000000boat255"),
    ("amz_0003", "boAt Rockerz 255 Pro+ Bluetooth Neckband", "Reliance Digital", 1399.0,
     "https://www.reliancedigital.in/boat-rockerz-255-pro/p/49300123"),
    # Samsung TV on 4 platforms -> best deal = Flipkart 31499
    ("amz_0004", "Samsung 43 inch Crystal 4K UHD Smart TV", "Amazon", 32999.0,
     "https://www.amazon.in/dp/B0SAMS43C4K"),
    ("amz_0004", "Samsung 43 inch Crystal 4K UHD Smart TV", "Flipkart", 31499.0,
     "https://www.flipkart.com/samsung-crystal-43/p/itmsamsung43c4k"),
    ("amz_0004", "Samsung 43 inch Crystal 4K UHD Smart TV", "Croma", 33990.0,
     "https://www.croma.com/samsung-crystal-43/p/251233"),
    ("amz_0004", "Samsung 43 inch Crystal 4K UHD Smart TV", "Reliance Digital", 32490.0,
     "https://www.reliancedigital.in/samsung-crystal-43/p/49301567"),
    # Prestige cooker on 3 platforms -> best deal = Flipkart 1149
    ("amz_0005", "Prestige Deluxe Alpha Stainless Steel Pressure Cooker", "Amazon", 1199.0,
     "https://www.amazon.in/dp/B0PRESTALPHA"),
    ("amz_0005", "Prestige Deluxe Alpha Stainless Steel Pressure Cooker", "Flipkart", 1149.0,
     "https://www.flipkart.com/prestige-deluxe-alpha/p/itmprestalpha"),
    ("amz_0005", "Prestige Deluxe Alpha Stainless Steel Pressure Cooker", "Croma", 1299.0,
     "https://www.croma.com/prestige-deluxe-alpha/p/233145"),
    # JBL headphones (Flipkart-origin id) on 3 platforms -> best deal = Amazon 2399
    ("fk_0001", "JBL Tune 510BT Wireless On-Ear Headphones", "Amazon", 2399.0,
     "https://www.amazon.in/dp/B0JBL510BT"),
    ("fk_0001", "JBL Tune 510BT Wireless On-Ear Headphones", "Flipkart", 2499.0,
     "https://www.flipkart.com/jbl-tune-510bt/p/itmjbl510bt"),
    ("fk_0001", "JBL Tune 510BT Wireless On-Ear Headphones", "Croma", 2599.0,
     "https://www.croma.com/jbl-tune-510bt/p/219876"),
    # Parle-G on 2 platforms (grocery lives only on Amazon/Flipkart) -> best = Amazon 10
    ("amz_0001", "Parle-G Original Glucose Biscuits", "Amazon", 10.0,
     "https://www.amazon.in/dp/B0PARLEG10"),
    ("amz_0001", "Parle-G Original Glucose Biscuits", "Flipkart", 12.0,
     "https://www.flipkart.com/parle-g-glucose/p/itmparleg10"),
    # Dove shampoo on a single platform -> exercises the no-comparison path (Req 7.5)
    ("amz_0006", "Dove Intense Repair Daily Shampoo", "Amazon", 299.0,
     "https://www.amazon.in/dp/B0DOVE340"),
]


def num(x):
    """Format a number without a spurious trailing ``.0`` for whole values."""
    x = round(float(x), 2)
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def round_price(x, T):
    """Round to a realistic granularity based on the category price scale."""
    if T >= 1000:
        return round(x, -1)   # nearest 10
    if T >= 100:
        return float(round(x))  # nearest 1
    return round(x, 1)        # nearest 0.1


def model_code(rng):
    return (rng.choice("ABDEHMNPRSTZ")
            + str(rng.randint(10, 99))
            + rng.choice("GLXSU"))


def make_row(cat, kind, rng):
    """Build one synthetic product row for a category with a discount pattern."""
    T = cat["T"]
    brand = rng.choice(cat["brands"])
    name = rng.choice(cat["names"])
    if kind == "genuine":
        # Discounted price genuinely BELOW the category norm; reference in the
        # normal band (roughly 13-33% off).
        discounted = round_price(rng.uniform(0.62, 0.92) * T, T)
        actual = round_price(discounted * rng.uniform(1.15, 1.5), T)
    else:  # inflated
        # Discounted price sits NEAR the norm; reference is a high outlier
        # (roughly 64-79% "off") -> a manufactured discount.
        discounted = round_price(rng.uniform(0.9, 1.15) * T, T)
        actual = round_price(discounted * rng.uniform(2.8, 4.8), T)
    if actual <= discounted:
        actual = round_price(discounted * 1.2, T)
    pct = round((actual - discounted) / actual * 100)
    rating = round(rng.uniform(cat["r_lo"], cat["r_hi"]), 1)
    rating_count = rng.randint(cat["rc_lo"], cat["rc_hi"])
    product_name = f"{brand} {name} {model_code(rng)}"
    return {
        "slug": cat["slug"],
        "amazon_cat": cat["amazon_cat"],
        "flipkart_tree": cat["flipkart_tree"],
        "brand": brand,
        "product_name": product_name,
        "discounted": discounted,
        "actual": actual,
        "discount_pct": pct,
        "rating": rating,
        "rating_count": rating_count,
        "kind": kind,
    }


def build():
    rng = random.Random(SEED)

    anchors_by_cat = {}
    for a in ANCHORS:
        anchors_by_cat.setdefault(a[2], []).append(a)

    amazon_rows, flipkart_rows = [], []
    amz_counter, fk_counter = 100, 100  # non-anchor ids start at *_0101

    for cat in CATEGORIES:
        slug = cat["slug"]
        cat_anchors = anchors_by_cat.get(slug, [])
        amazon_anchor_ct = sum(1 for a in cat_anchors if a[3] == "amazon")
        flipkart_anchor_ct = sum(1 for a in cat_anchors if a[3] == "flipkart")

        # Target 22 rows/category: 14 genuine (incl. anchors) + 8 inflated.
        n_genuine_random = 14 - len(cat_anchors)
        n_inflated_random = 8
        generated = [make_row(cat, "genuine", rng) for _ in range(n_genuine_random)]
        generated += [make_row(cat, "inflated", rng) for _ in range(n_inflated_random)]
        rng.shuffle(generated)

        # Fill each file up to its per-category quota (13 amazon / 9 flipkart).
        amazon_need = 13 - amazon_anchor_ct
        flipkart_need = 9 - flipkart_anchor_ct
        amazon_slice = generated[:amazon_need]
        flipkart_slice = generated[amazon_need:amazon_need + flipkart_need]

        # Anchors first (fixed ids), then generated rows (assigned ids).
        for a in cat_anchors:
            aid, aname, aslug, afile, adisc, aact, abrand, arat, arc = a
            pct = round((aact - adisc) / aact * 100)
            row = {
                "slug": aslug, "amazon_cat": cat["amazon_cat"],
                "flipkart_tree": cat["flipkart_tree"], "brand": abrand,
                "product_name": aname, "discounted": adisc, "actual": aact,
                "discount_pct": pct, "rating": arat, "rating_count": arc,
                "kind": "genuine", "product_id": aid,
            }
            (amazon_rows if afile == "amazon" else flipkart_rows).append(row)

        for row in amazon_slice:
            amz_counter += 1
            row["product_id"] = f"amz_{amz_counter:04d}"
            amazon_rows.append(row)
        for row in flipkart_slice:
            fk_counter += 1
            row["product_id"] = f"fk_{fk_counter:04d}"
            flipkart_rows.append(row)

    return amazon_rows, flipkart_rows


def pick_messy_indices(amazon_rows):
    """Choose rows to render with Rs. symbols / commas / percent signs so the
    ingestion cleaning logic has both genuine and inflated messy rows to strip.
    Returns a set of indices: first 3 inflated + 3 spaced genuine rows."""
    inflated_idx = [i for i, r in enumerate(amazon_rows) if r["kind"] == "inflated"]
    genuine_idx = [i for i, r in enumerate(amazon_rows) if r["kind"] == "genuine"]
    messy = set(inflated_idx[:3])
    if genuine_idx:
        step = max(1, len(genuine_idx) // 3)
        messy.update(genuine_idx[::step][:3])
    return messy


def write_amazon(amazon_rows):
    messy = pick_messy_indices(amazon_rows)
    path = OUT_DIR / "amazon_sample.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["product_id", "product_name", "category", "discounted_price",
                    "actual_price", "discount_percentage", "rating", "rating_count"])
        for i, r in enumerate(amazon_rows):
            if i in messy:
                disc = f"\u20b9{r['discounted']:,.0f}"
                act = f"\u20b9{r['actual']:,.0f}"
                pct = f"{r['discount_pct']}%"
                rc = f"{r['rating_count']:,}"
            else:
                disc = num(r["discounted"])
                act = num(r["actual"])
                pct = str(r["discount_pct"])
                rc = str(r["rating_count"])
            w.writerow([r["product_id"], r["product_name"], r["amazon_cat"],
                        disc, act, pct, num(r["rating"]), rc])
    return path, len(amazon_rows), len(messy)


def write_flipkart(flipkart_rows):
    path = OUT_DIR / "flipkart_sample.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["product_id", "product_name", "category_tree", "retail_price",
                    "discounted_price", "brand", "overall_rating", "rating_count"])
        for r in flipkart_rows:
            w.writerow([r["product_id"], r["product_name"], r["flipkart_tree"],
                        num(r["actual"]), num(r["discounted"]), r["brand"],
                        num(r["rating"]), str(r["rating_count"])])
    return path, len(flipkart_rows)


def write_pack_size_history():
    path = OUT_DIR / "pack_size_history.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["product_id", "product_name", "observed_at", "pack_quantity",
                    "pack_unit", "selling_price", "source_type", "source_citation"])
        for row in PACK_SIZE_HISTORY:
            pid, pname, obs, qty, unit, price, stype, cite = row
            w.writerow([pid, pname, obs, num(qty), unit, num(price), stype, cite])
    return path, len(PACK_SIZE_HISTORY)


def write_platform_prices():
    path = OUT_DIR / "platform_prices.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["product_id", "product_name", "platform", "price", "product_url"])
        for row in PLATFORM_PRICES:
            pid, pname, platform, price, url = row
            w.writerow([pid, pname, platform, num(price), url])
    return path, len(PLATFORM_PRICES)


def main():
    amazon_rows, flipkart_rows = build()
    a_path, a_n, a_messy = write_amazon(amazon_rows)
    f_path, f_n = write_flipkart(flipkart_rows)
    p_path, p_n = write_pack_size_history()
    pp_path, pp_n = write_platform_prices()

    print("Synthetic fixtures written to", OUT_DIR)
    print(f"  amazon_sample.csv       : {a_n} rows ({a_messy} rows with Rs./comma/% formatting)")
    print(f"  flipkart_sample.csv     : {f_n} rows")
    print(f"  pack_size_history.csv   : {p_n} rows")
    print(f"  platform_prices.csv     : {pp_n} rows")
    print(f"  amazon + flipkart total : {a_n + f_n} product rows")


if __name__ == "__main__":
    main()
