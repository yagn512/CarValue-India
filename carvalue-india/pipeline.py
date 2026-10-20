"""
CarValue India - data + training pipeline.

Run:  python pipeline.py

What it does
1. Gets a REAL Indian used-car dataset (CarDekho) -> data/cars.csv
2. Cleans the data (duplicates, missing values, impossible values, price outliers)
3. Trains exactly 3 models: Linear Regression, Decision Tree, Random Forest
   (80% train / 20% test, random_state=42)
4. Saves MAE, MSE, RMSE, R2, the best model, charts and metadata
"""
import glob
import json
import os
import sys
import urllib.request
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
CHART_DIR = os.path.join(BASE, "static", "charts")
CSV_PATH = os.environ.get("CARVALUE_CSV", os.path.join(DATA_DIR, "cars.csv"))
CURRENT_YEAR = datetime.now().year

# Public Kaggle datasets that contain CarDekho used-car listings.
KAGGLE_HANDLES = [
    "nehalbirla/vehicle-dataset-from-cardekho",
    "manishkr1754/cardekho-used-car-data",
]
# Optional: set CARVALUE_DATA_URL to any direct link to a CarDekho CSV.
EXTRA_URL = os.environ.get("CARVALUE_DATA_URL")

NAVY = "#0b1f3a"
BLUE = "#2f6fdb"


# ----------------------------------------------------------------- 1. DATA
def _valid_csv(path):
    try:
        cols = [c.strip().lower() for c in pd.read_csv(path, nrows=1).columns]
    except Exception:
        return False
    return "selling_price" in cols and "km_driven" in cols


def _best_csv_in(folder):
    found = [p for p in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True) if _valid_csv(p)]
    # prefer the file that has a plain "year" column (matches the website form)
    def score(p):
        cols = [c.strip().lower() for c in pd.read_csv(p, nrows=1).columns]
        return ("year" in cols, os.path.getsize(p))
    return max(found, key=score) if found else None


def get_dataset():
    """Return (dataframe, source_description). Never creates fake data."""
    if os.path.exists(CSV_PATH) and _valid_csv(CSV_PATH):
        print(f"Using existing dataset: {CSV_PATH}")
        return pd.read_csv(CSV_PATH), "CarDekho used-car dataset (local data/cars.csv)"

    if EXTRA_URL:
        try:
            print(f"Downloading {EXTRA_URL} ...")
            urllib.request.urlretrieve(EXTRA_URL, os.path.join(DATA_DIR, "cars.csv"))
            if _valid_csv(os.path.join(DATA_DIR, "cars.csv")):
                return pd.read_csv(os.path.join(DATA_DIR, "cars.csv")), f"CarDekho dataset ({EXTRA_URL})"
        except Exception as exc:
            print("  URL download failed:", exc)

    try:
        import kagglehub
    except ImportError:
        kagglehub = None
    if kagglehub:
        for handle in KAGGLE_HANDLES:
            try:
                print(f"Downloading Kaggle dataset {handle} ...")
                folder = kagglehub.dataset_download(handle)
                path = _best_csv_in(folder)
                if path:
                    df = pd.read_csv(path)
                    df.to_csv(os.path.join(DATA_DIR, "cars.csv"), index=False)
                    return df, f"Kaggle: {handle} ({os.path.basename(path)})"
            except Exception as exc:
                print("  failed:", exc)

    sys.exit(
        "\nERROR: could not download the dataset automatically.\n"
        "Fix: put a real CarDekho CSV (columns like name, year, selling_price, km_driven,\n"
        "fuel, seller_type, transmission) at data/cars.csv, OR set CARVALUE_DATA_URL to a\n"
        "direct CSV link, then run  python pipeline.py  again. See README.md."
    )


def normalize(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"fuel_type": "fuel", "transmission_type": "transmission"})
    if "brand" not in df.columns:
        df["brand"] = df["name"].astype(str).str.split().str[0]
    df["brand"] = df["brand"].astype(str).str.strip().str.title().replace({"Land": "Land Rover", "Bmw": "BMW", "Mg": "MG", "Ambassador": "Ambassador"})
    if "year" in df.columns:
        df["car_age"] = CURRENT_YEAR - pd.to_numeric(df["year"], errors="coerce")
    elif "vehicle_age" in df.columns:
        df["car_age"] = pd.to_numeric(df["vehicle_age"], errors="coerce")
    else:
        sys.exit("Dataset has neither a 'year' nor a 'vehicle_age' column.")
    for col in ["fuel", "transmission", "seller_type"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    return df[["brand", "car_age", "km_driven", "fuel", "transmission", "seller_type", "selling_price"]]


def clean(df):
    log = {"rows_raw": int(len(df))}
    df = df.dropna()
    log["after_missing_values"] = int(len(df))
    df = df.drop_duplicates()
    log["after_duplicates"] = int(len(df))
    df = df[(df.selling_price > 0) & (df.km_driven >= 0) & (df.car_age >= 0)]
    q1, q3 = df.selling_price.quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)  # only extreme price outliers are removed
    df = df[df.selling_price <= upper]
    log["after_outliers"] = int(len(df))
    counts = df.brand.value_counts()
    df = df[df.brand.isin(counts[counts >= 10].index)]  # drop very rare brands
    log["rows_final"] = int(len(df))
    return df.reset_index(drop=True), log


# --------------------------------------------------------------- 2. CHARTS
def make_charts(df, metrics):
    os.makedirs(CHART_DIR, exist_ok=True)

    def style(ax, title, xl="", yl=""):
        ax.set_title(title, color=NAVY, fontweight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)

    top = df.brand.value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(top.index, top.values, color=NAVY)
    style(ax, "Top 10 Brands by Number of Cars", "Brand", "Cars")
    plt.xticks(rotation=40, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "brands.png"), dpi=110); plt.close(fig)

    fuel = df.fuel.value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fuel.index, fuel.values, color=BLUE)
    style(ax, "Cars by Fuel Type", "Fuel type", "Cars")
    fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "fuel.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.selling_price / 1e5, bins=40, color=NAVY, edgecolor="white")
    style(ax, "Distribution of Selling Prices", "Selling price (₹ lakh)", "Cars")
    fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "price.png"), dpi=110); plt.close(fig)

    avg = (df.groupby("brand").selling_price.mean() / 1e5).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(avg.index, avg.values, color=BLUE)
    style(ax, "Average Selling Price by Brand (Top 10)", "Brand", "₹ lakh")
    plt.xticks(rotation=40, ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "avg_price_brand.png"), dpi=110); plt.close(fig)

    names = [m["model"] for m in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [m["r2"] for m in metrics], color=[NAVY, BLUE, "#6c8ebf"])
    style(ax, "R² Score of the Three Models (test set)", "", "R²")
    fig.tight_layout(); fig.savefig(os.path.join(CHART_DIR, "model_r2.png"), dpi=110); plt.close(fig)

    return {
        "top_brands": {k: int(v) for k, v in top.items()},
        "fuel_counts": {k: int(v) for k, v in fuel.items()},
        "avg_price_by_brand_lakh": {k: round(float(v), 2) for k, v in avg.items()},
        "price_summary_lakh": {k: round(float(v) / 1e5, 2) for k, v in df.selling_price.describe().items() if k != "count"},
    }


# ---------------------------------------------------------------- 3. MODELS
def main():
    for d in (DATA_DIR, MODEL_DIR, CHART_DIR):
        os.makedirs(d, exist_ok=True)

    raw, source = get_dataset()
    df, log = clean(normalize(raw))
    print("Cleaning summary:", log)

    X = df.drop(columns="selling_price")
    y = df["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat = ["brand", "fuel", "transmission", "seller_type"]
    num = ["car_age", "km_driven"]
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", "passthrough", num)]
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(min_samples_leaf=3, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
    }

    metrics, fitted = [], {}
    for name, reg in models.items():
        pipe = Pipeline([("prep", prep), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        row = {
            "model": name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, pred)),
        }
        metrics.append(row)
        fitted[name] = pipe
        print(f"{name:26s} MAE={row['mae']:.0f}  RMSE={row['rmse']:.0f}  R2={row['r2']:.4f}")

    best = min(metrics, key=lambda m: m["rmse"])  # lowest RMSE = best model
    print("Best model:", best["model"])
    joblib.dump(fitted[best["model"]], os.path.join(MODEL_DIR, "best_model.joblib"), compress=3)

    agg = make_charts(df, metrics)
    meta = {
        "dataset_source": source,
        "cleaning": log,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split": "80% train / 20% test, random_state=42",
        "metrics": metrics,
        "best_model": best["model"],
        "best_mae": best["mae"],
        "reference_year": CURRENT_YEAR,
        "options": {
            "brands": sorted(df.brand.unique().tolist()),
            "fuels": sorted(df.fuel.unique().tolist()),
            "transmissions": sorted(df.transmission.unique().tolist()),
            "seller_types": sorted(df.seller_type.unique().tolist()),
        },
        "aggregates": agg,
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Done. Now run:  python app.py")


if __name__ == "__main__":
    main()
