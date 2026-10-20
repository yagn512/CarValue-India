"""CarValue India - Flask app.  Run:  python app.py   (after python pipeline.py)"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "best_model.joblib")
META_PATH = os.path.join(BASE, "models", "metadata.json")
CURRENT_YEAR = datetime.now().year

app = Flask(__name__)
_state = {}


def inr(value):
    """Format a number as Indian-style rupees, e.g. 450000 -> ₹4,50,000."""
    n = str(int(round(value)))
    if len(n) > 3:
        head, tail = n[:-3], n[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        n = ",".join(parts + [tail])
    return "₹" + n


def load():
    """Load model + metadata once. Returns None if pipeline.py was not run yet."""
    if "model" not in _state:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            return None
        _state["model"] = joblib.load(MODEL_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _state["meta"] = json.load(f)
    return _state


def estimate(data):
    st = load()
    if st is None:
        raise ValueError("Model not trained yet. Run: python pipeline.py")
    opts = st["meta"]["options"]
    try:
        year = int(data.get("year"))
        km = float(data.get("km_driven"))
    except (TypeError, ValueError):
        raise ValueError("Please enter a valid year and kilometres driven.")
    if not 1990 <= year <= CURRENT_YEAR:
        raise ValueError(f"Year must be between 1990 and {CURRENT_YEAR}.")
    if km < 0 or km > 1_000_000:
        raise ValueError("Kilometres driven must be between 0 and 10,00,000.")
    for key, field in [("brand", "brands"), ("fuel", "fuels"),
                       ("transmission", "transmissions"), ("seller_type", "seller_types")]:
        if data.get(key) not in opts[field]:
            raise ValueError(f"Please choose a valid {key.replace('_', ' ')}.")
    row = pd.DataFrame([{
        "brand": data["brand"], "car_age": CURRENT_YEAR - year, "km_driven": km,
        "fuel": data["fuel"], "transmission": data["transmission"],
        "seller_type": data["seller_type"],
    }])
    price = max(float(st["model"].predict(row)[0]), 0)
    mae = st["meta"]["best_mae"]  # typical error of the best model on the test set
    low, high = max(price - mae, 0), price + mae
    return {"price": round(price), "price_text": inr(price),
            "low_text": inr(low), "high_text": inr(high), "mae_text": inr(mae)}


@app.route("/")
def home():
    return render_template("index.html", meta=(load() or {}).get("meta"))


@app.route("/predict")
def predict_page():
    return render_template("predict.html", meta=(load() or {}).get("meta"), year_now=CURRENT_YEAR)


@app.route("/compare")
def compare_page():
    return render_template("compare.html", meta=(load() or {}).get("meta"), year_now=CURRENT_YEAR)


@app.route("/insights")
def insights_page():
    return render_template("insights.html", meta=(load() or {}).get("meta"))


@app.route("/performance")
def performance_page():
    return render_template("performance.html", meta=(load() or {}).get("meta"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(ok=True, **estimate(request.get_json(force=True)))
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400


@app.route("/api/compare", methods=["POST"])
def api_compare():
    body = request.get_json(force=True)
    try:
        a, b = estimate(body.get("car_a", {})), estimate(body.get("car_b", {}))
    except ValueError as exc:
        return jsonify(ok=False, error=str(exc)), 400
    diff = abs(a["price"] - b["price"])
    if diff == 0:
        verdict = "Both cars have the same estimated price."
    else:
        cheaper = "Car A" if a["price"] < b["price"] else "Car B"
        verdict = f"{cheaper} is estimated to be cheaper by {inr(diff)}."
    return jsonify(ok=True, car_a=a, car_b=b, verdict=verdict)


@app.template_filter("inr")
def inr_filter(v):
    return inr(v)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
