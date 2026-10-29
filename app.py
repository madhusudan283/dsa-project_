from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file

from sustainability import (
    FEATURE_LABELS,
    INDICATORS,
    binary_search_ids,
    create_chart,
    get_bundle,
    merge_sort_records,
)

app = Flask(__name__, static_folder="static", static_url_path="")


def state() -> dict[str, Any]:
    return get_bundle()


def json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    try:
        bundle = state()
    except RuntimeError as error:
        return json_error(str(error), 503)
    return jsonify({"status": "ready", "records": len(bundle["data"])})


@app.get("/api/overview")
def overview():
    try:
        bundle = state()
    except RuntimeError as error:
        return json_error(str(error), 503)
    data = bundle["data"]
    feature_names = bundle["features"]
    numeric = data[feature_names + ["NEXT_GDP_GROWTH"]]
    grouped = data.groupby("OUTCOME", observed=True).agg(
        records=("ID", "count"),
        mean_growth=("NEXT_GDP_GROWTH", "mean"),
        mean_renewables=("RENEWABLE_SHARE", "mean"),
        mean_life_expectancy=("LIFE_EXPECTANCY", "mean"),
    ).reset_index()
    array_features = [name for name in ("GDP_GROWTH", "RENEWABLE_SHARE", "LIFE_EXPECTANCY") if name in feature_names]
    array = data[array_features].fillna(data[array_features].median()).to_numpy(dtype=float)
    calibration_matrix = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    calibration_values = np.array([5, 6, 5], dtype=float)
    return jsonify(
        {
            "records": len(data),
            "countries": int(data["country_code"].nunique()),
            "regions": sorted(data["region"].dropna().unique().tolist()),
            "years": [int(data["year"].min()), int(data["year"].max())],
            "features": feature_names,
            "feature_labels": {name: FEATURE_LABELS[name] for name in feature_names},
            "target": "NEXT_GDP_GROWTH",
            "target_median": bundle["target_median"],
            "missing_values": int(data[feature_names].isna().sum().sum()),
            "duplicate_rows": int(data.duplicated(["country_code", "year"]).sum()),
            "preprocessing": bundle["preprocessing"],
            "numpy_lab": {
                "shape": list(array.shape),
                "indexed_sample": array[0].round(2).tolist(),
                "slice_preview": array[:2].round(2).tolist(),
                "reshape": list(array[:12].reshape(4, 3, 3).shape),
                "vectorized_mean": array.mean(axis=0).round(2).tolist(),
                "matrix_solution": np.linalg.solve(calibration_matrix, calibration_values).round(2).tolist(),
                "array_features": array_features,
            },
            "summary": {
                "mean": numeric.mean().round(3).to_dict(),
                "median": numeric.median().round(3).to_dict(),
                "variance": numeric.var().round(3).to_dict(),
                "std": numeric.std().round(3).to_dict(),
                "correlation": numeric.corr().round(3).to_dict(),
            },
            "groups": grouped.round(2).to_dict(orient="records"),
            "metrics": bundle["metrics"],
        }
    )


@app.get("/api/countries")
def countries():
    try:
        data = state()["data"]
    except RuntimeError as error:
        return json_error(str(error), 503)
    catalog = data[["country_code", "country", "region"]].drop_duplicates().sort_values("country")
    return jsonify(catalog.to_dict(orient="records"))


@app.get("/api/series/<country_code>")
def country_series(country_code: str):
    try:
        data = state()["data"]
    except RuntimeError as error:
        return json_error(str(error), 503)
    rows = data[data["country_code"].str.upper() == country_code.upper()].sort_values("year")
    if rows.empty:
        return json_error("Country was not found in the dataset.", 404)
    return jsonify(
        {
            "country": rows.iloc[0]["country"],
            "region": rows.iloc[0]["region"],
            "records": rows.replace({np.nan: None}).to_dict(orient="records"),
        }
    )


@app.get("/api/records")
def records():
    try:
        bundle = state()
    except RuntimeError as error:
        return json_error(str(error), 503)
    data = bundle["data"]
    available = set(data.columns)
    requested_feature = request.args.get("feature", "GDP_GROWTH").upper()
    if requested_feature not in available:
        return json_error("Unknown sort or filter feature.")
    query = request.args.get("q", "").strip().casefold()
    direction = request.args.get("direction", "desc")
    region = request.args.get("region", "").strip()
    try:
        minimum = float(request.args["min"]) if request.args.get("min") else None
        maximum = float(request.args["max"]) if request.args.get("max") else None
        page = max(1, int(request.args.get("page", "1")))
        page_size = min(50, max(1, int(request.args.get("page_size", "12"))))
    except ValueError:
        return json_error("Invalid numeric filter or page.")

    selected = data
    if minimum is not None:
        selected = selected[pd.to_numeric(selected[requested_feature], errors="coerce") >= minimum]
    if maximum is not None:
        selected = selected[pd.to_numeric(selected[requested_feature], errors="coerce") <= maximum]
    if region:
        selected = selected[selected["region"] == region]
    if query:
        mask = selected.astype(str).apply(lambda column: column.str.casefold().str.contains(query, regex=False)).any(axis=1)
        selected = selected[mask]
    row_records = selected.replace({np.nan: None}).to_dict(orient="records")
    row_records = merge_sort_records(row_records, requested_feature, direction != "asc")
    start = (page - 1) * page_size
    return jsonify(
        {
            "records": row_records[start : start + page_size],
            "total": len(row_records),
            "page": page,
            "page_size": page_size,
        }
    )


@app.get("/api/record/<int:record_id>")
def get_record(record_id: int):
    try:
        bundle = state()
    except RuntimeError as error:
        return json_error(str(error), 503)
    position = binary_search_ids(bundle["record_ids"], record_id)
    if position < 0:
        return json_error("Record not found.", 404)
    record = bundle["data"].iloc[position].replace({np.nan: None}).to_dict()
    return jsonify(record)


@app.post("/api/predict")
def predict():
    try:
        bundle = state()
    except RuntimeError as error:
        return json_error(str(error), 503)
    payload = request.get_json(silent=True) or {}
    values = {}
    for name in bundle["features"]:
        value = payload.get(name.lower(), payload.get(name))
        if value in (None, ""):
            values[name] = np.nan
            continue
        try:
            values[name] = float(value)
        except (TypeError, ValueError):
            return json_error(f"{FEATURE_LABELS[name]} must be a number.")
    input_frame = pd.DataFrame([values], columns=bundle["features"])
    forecast = float(bundle["regression"].predict(input_frame)[0])
    probabilities = bundle["classifier"].predict_proba(input_frame)[0]
    classes = list(bundle["classifier"].classes_)
    above_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    return jsonify(
        {
            "next_year": int(bundle["data"]["year"].max()) + 1,
            "forecast_growth": round(forecast, 2),
            "above_typical_probability": round(above_probability, 4),
            "predicted_group": "Above typical" if above_probability >= 0.5 else "Below typical",
            "typical_growth_cutoff": round(bundle["target_median"], 2),
        }
    )


@app.get("/api/chart/<chart_type>.png")
def chart(chart_type: str):
    try:
        bundle = state()
    except RuntimeError as error:
        return json_error(str(error), 503)
    try:
        image = create_chart(bundle["data"], bundle["features"], chart_type)
    except ValueError as error:
        return json_error(str(error), 404)
    return send_file(image, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
