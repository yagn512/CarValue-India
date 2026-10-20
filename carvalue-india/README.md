# CarValue India – Used Car Price Prediction

A Data Analytics & Data Science mini project (BVM, 342IT). Predicts the selling price of an
Indian used car in ₹ using machine learning trained on real CarDekho listings.

**Stack:** Python, Flask, Pandas, NumPy, Scikit-learn, Matplotlib, HTML/CSS/JavaScript, Gunicorn (Render).

## Pages
Home · Price Prediction (with approximate range) · Compare Cars · Car Insights (real-data charts) · Model Performance (MAE, MSE, RMSE, R²).

## Models
Exactly three: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
80% train / 20% test, `random_state=42`. The best model is the one with the lowest RMSE on the test set.
The price range shown on the website is *predicted price ± the best model's MAE* and is only an estimate.

## Project structure
```
app.py            Flask website + JSON API
pipeline.py       download dataset -> clean -> train 3 models -> save best model, charts, metadata
requirements.txt  Python packages
Procfile          gunicorn app:app   (Render / Heroku-style hosting)
render.yaml       optional Render blueprint
data/             cars.csv is saved here by pipeline.py
models/           best_model.joblib + metadata.json (created by pipeline.py)
templates/        HTML pages
static/           css, js, charts (charts are created by pipeline.py)
```

## Run locally (any terminal inside the project folder)
```
pip install -r requirements.txt
python pipeline.py
python app.py
```
Open http://127.0.0.1:5000 (in a cloud workspace, open the forwarded/preview link for port 5000).

`python pipeline.py` downloads the real CarDekho dataset (via Kaggle's public dataset download),
trains the models and prints the metrics. It needs internet access. Run it again any time to retrain.

## Dataset
CarDekho used-car listings (public Kaggle datasets). No fake or generated data is used.
If the automatic download fails, either:
- put any real CarDekho CSV at `data/cars.csv` (needs columns such as `name, year, selling_price, km_driven, fuel, seller_type, transmission`), or
- set a direct CSV link: `CARVALUE_DATA_URL=<link>` and run `python pipeline.py` again.

## Deploy on Render
1. Push this folder to a GitHub repository.
2. Render dashboard → **New +** → **Web Service** → connect the repository.
3. Runtime **Python 3**, Build Command: `pip install -r requirements.txt && python pipeline.py`,
   Start Command: `gunicorn app:app`. (Or choose **Blueprint** and use `render.yaml`.)
4. Click **Create Web Service**. After the build finishes, open the `https://<name>.onrender.com` URL.
   The free plan sleeps when idle, so the first visit can take about a minute.

## Common errors
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | run `pip install -r requirements.txt` |
| Yellow "model has not been trained" banner | run `python pipeline.py`, then restart `python app.py` |
| Dataset download fails | see **Dataset** above |
| `Address already in use` | `PORT=5001 python app.py` |
| Render build fails on the dataset | commit `data/cars.csv` to GitHub, or set `CARVALUE_DATA_URL` in Render → Environment |
