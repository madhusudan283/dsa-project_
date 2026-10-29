# Fieldnotes: Sustainable Growth Lab

A local dashboard for exploring how country-level economic growth relates to development and environmental indicators. The app uses annual country data from the World Bank Indicators API and predicts next-year GDP growth from the current year's available indicators.

This is an educational forecasting project, not an official economic forecast, a sustainability rating, or a policy recommendation.

## Run locally

Python 3.10 or newer is recommended. The first run needs an internet connection to fetch World Bank indicators. The app caches the fetched country-year panel in `data/country_sustainability.csv` for later offline starts.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. To run without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

The attached C elevator simulator is available as a standalone browser simulation at http://127.0.0.1:5000/elevator.html. It recreates the 0–20 floor controls, normal and priority requests, and direction-aware dispatch without requiring a C compiler.

## Deploy the elevator frontend to GitHub Pages

The elevator simulator is a standalone static frontend. A GitHub Actions workflow publishes only that frontend to GitHub Pages; the Python/Flask API and sustainability dashboard are not included in the Pages deployment.

1. In the repository settings, open **Pages** and set the build and deployment source to **GitHub Actions**.
2. Push the workflow and frontend files to the `main` branch, or run **Deploy elevator frontend to GitHub Pages** from the **Actions** tab.
3. Open the Pages URL shown in the completed workflow deployment.

## Data and model

Indicators are fetched from the World Bank API for 2000 onward, including GDP growth, GDP per capita, renewable energy share, electricity access, life expectancy, and forest area. Countries are joined by ISO-3 code. The loader removes regional aggregate rows, aligns each observation with the same country's next calendar-year GDP growth, and caches the result locally.

The regression target is next-year GDP growth (annual percent). A separate logistic classifier predicts whether that growth is above or below the median in the training years. Models use a chronological split: earlier years train the models and later years evaluate them. Missing input indicators are imputed from training data inside the scikit-learn pipeline.

## Features

- NumPy indexing, slicing, reshaping, vectorized summaries, and matrix solving.
- Pandas cleaning, descriptive statistics, correlation, group-by summaries, country and region filtering, and pagination.
- Matplotlib trend, regional comparison, histogram, scatter, and correlation heatmap charts.
- Linear regression and logistic classification for next-year GDP growth.
- Custom merge sort for table ordering (O(n log n)) and binary search for cached record-ID lookup (O(log n)).
- Interactive country/region table and scenario prediction form.

## API

- `GET /api/overview`
- `GET /api/records?feature=GDP_GROWTH&region=Europe%20%26%20Central%20Asia&page=1`
- `GET /api/series/USA`
- `GET /api/record/1`
- `POST /api/predict` with indicator values in JSON
- `GET /api/chart/line.png` (also `bar`, `histogram`, `scatter`, `heatmap`)

## Tests

```powershell
python -m unittest -v
```

Tests use a small deterministic fixture and do not require internet access. On first app startup, the public World Bank API must be reachable to create the cached dataset. The frontend loads its display fonts from Google Fonts when online; the system fallback fonts work offline.
