from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
CACHE_FILE = ROOT / "data" / "country_sustainability.csv"
WORLD_BANK_API = "https://api.worldbank.org/v2"
START_YEAR = 2000
END_YEAR = 2024
INDICATORS = {
    "GDP_GROWTH": "NY.GDP.MKTP.KD.ZG",
    "GDP_PER_CAPITA": "NY.GDP.PCAP.CD",
    "RENEWABLE_SHARE": "EG.FEC.RNEW.ZS",
    "ELECTRICITY_ACCESS": "EG.ELC.ACCS.ZS",
    "LIFE_EXPECTANCY": "SP.DYN.LE00.IN",
    "FOREST_SHARE": "AG.LND.FRST.ZS",
}
FEATURE_LABELS = {
    "GDP_GROWTH": "GDP growth (%)",
    "GDP_PER_CAPITA": "Income per person (US$)",
    "RENEWABLE_SHARE": "Renewable energy (%)",
    "ELECTRICITY_ACCESS": "Electricity access (%)",
    "LIFE_EXPECTANCY": "Life expectancy (years)",
    "FOREST_SHARE": "Forest area (%)",
    "NEXT_GDP_GROWTH": "Next-year GDP growth (%)",
}


def _get_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "Fieldnotes-Sustainability-Dashboard/1.0"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _fetch_country_catalog() -> dict[str, dict[str, str]]:
    payload = _get_json(f"{WORLD_BANK_API}/country?{urlencode({'format': 'json', 'per_page': 400})}")
    catalog = {}
    for item in payload[1] or []:
        code = item.get("iso2Code")
        iso3 = item.get("id")
        region = item.get("region") or {}
        if code and iso3 and region.get("id") not in (None, "", "NA"):
            catalog[iso3] = {"country": item["name"], "region": region["value"]}
    return catalog


def _fetch_indicator(args: tuple[str, str]) -> list[dict[str, Any]]:
    feature, code = args
    query = urlencode({"date": f"{START_YEAR}:{END_YEAR}", "format": "json", "per_page": 20000})
    payload = _get_json(f"{WORLD_BANK_API}/country/all/indicator/{code}?{query}")
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return []
    rows = []
    for item in payload[1]:
        iso3 = item.get("countryiso3code")
        value = item.get("value")
        if iso3 and value is not None:
            rows.append(
                {
                    "country_code": iso3,
                    "year": int(item["date"]),
                    "feature": feature,
                    "value": float(value),
                }
            )
    return rows


def download_world_bank_data() -> pd.DataFrame:
    catalog = _fetch_country_catalog()
    with ThreadPoolExecutor(max_workers=len(INDICATORS)) as executor:
        result_lists = list(executor.map(_fetch_indicator, INDICATORS.items()))
    observations = pd.DataFrame(row for rows in result_lists for row in rows)
    if observations.empty or "GDP_GROWTH" not in set(observations["feature"]):
        raise RuntimeError("The World Bank API did not return GDP growth data.")

    observations = observations[observations["country_code"].isin(catalog)]
    wide = observations.pivot_table(
        index=["country_code", "year"], columns="feature", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    for feature in INDICATORS:
        if feature not in wide:
            wide[feature] = np.nan
    wide["country"] = wide["country_code"].map(lambda code: catalog[code]["country"])
    wide["region"] = wide["country_code"].map(lambda code: catalog[code]["region"])
    wide = wide.sort_values(["country_code", "year"]).reset_index(drop=True)
    target_lookup = wide.set_index(["country_code", "year"])["GDP_GROWTH"]
    next_year_keys = pd.MultiIndex.from_arrays(
        [wide["country_code"], wide["year"] + 1], names=["country_code", "year"]
    )
    wide["NEXT_GDP_GROWTH"] = target_lookup.reindex(next_year_keys).to_numpy()
    wide = wide.dropna(subset=["NEXT_GDP_GROWTH"]).reset_index(drop=True)
    if len(wide) < 100:
        raise RuntimeError("The World Bank API returned too few usable country-year records.")
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(CACHE_FILE, index=False)
    return wide


def load_world_bank_data() -> pd.DataFrame:
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE)
        if len(cached) >= 100 and "NEXT_GDP_GROWTH" in cached:
            return cached
    try:
        return download_world_bank_data()
    except (TimeoutError, URLError, OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "Could not load country indicators from the World Bank API. Check your internet connection and restart the app."
        ) from error


def merge_sort_records(records: list[dict[str, Any]], key: str, descending: bool = False) -> list[dict[str, Any]]:
    """Sort records with merge sort in O(n log n) time and O(n) extra space."""
    if len(records) < 2:
        return records
    middle = len(records) // 2
    left = merge_sort_records(records[:middle], key, descending)
    right = merge_sort_records(records[middle:], key, descending)
    merged: list[dict[str, Any]] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_value, right_value = left[left_index].get(key), right[right_index].get(key)
        if left_value is None:
            take_left = False
        elif right_value is None:
            take_left = True
        else:
            take_left = left_value >= right_value if descending else left_value <= right_value
        if take_left:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    return merged + left[left_index:] + right[right_index:]


def binary_search_ids(sorted_ids: list[int], target_id: int) -> int:
    """Return the matching index in a sorted ID list, or -1 when absent."""
    low, high = 0, len(sorted_ids) - 1
    while low <= high:
        middle = (low + high) // 2
        if sorted_ids[middle] == target_id:
            return middle
        if sorted_ids[middle] < target_id:
            low = middle + 1
        else:
            high = middle - 1
    return -1


def prepare_dataset(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    feature_names = [name for name in INDICATORS if name in raw_data and raw_data[name].notna().any()]
    data = raw_data.copy()
    audit = {
        "input_rows": int(len(data)),
        "missing_before": int(data[feature_names].isna().sum().sum()),
        "duplicates_removed": int(data.duplicated(["country_code", "year"]).sum()),
    }
    data = data.drop_duplicates(["country_code", "year"]).reset_index(drop=True)
    data[feature_names] = data[feature_names].apply(pd.to_numeric, errors="coerce")
    audit["missing_after"] = int(data[feature_names].isna().sum().sum())
    data.insert(0, "ID", np.arange(1, len(data) + 1))
    target_median = float(data["NEXT_GDP_GROWTH"].median())
    data["OUTCOME"] = np.where(data["NEXT_GDP_GROWTH"] >= target_median, "Above typical", "Below typical")
    audit["imputed_by_model"] = int(data[feature_names].isna().sum().sum())
    audit["rows_retained"] = int(len(data))
    return data, feature_names, audit


def train_models(data: pd.DataFrame, feature_names: list[str]) -> tuple[Any, Any, dict[str, float], float]:
    ordered_years = sorted(int(year) for year in data["year"].unique())
    cutoff_index = max(1, int(len(ordered_years) * 0.8))
    cutoff_year = ordered_years[min(cutoff_index, len(ordered_years) - 1)]
    train_mask = data["year"] < cutoff_year
    test_mask = ~train_mask
    if train_mask.sum() < 30 or test_mask.sum() < 10:
        cutoff_year = int(data["year"].quantile(0.8))
        train_mask = data["year"] < cutoff_year
        test_mask = ~train_mask
    X_train, X_test = data.loc[train_mask, feature_names], data.loc[test_mask, feature_names]
    y_train = data.loc[train_mask, "NEXT_GDP_GROWTH"]
    y_test = data.loc[test_mask, "NEXT_GDP_GROWTH"]
    target_median = float(y_train.median())

    regression = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    regression.fit(X_train, y_train)
    regression_predictions = regression.predict(X_test)

    train_classes = (y_train >= target_median).astype(int)
    test_classes = (y_test >= target_median).astype(int)
    classifier = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
    )
    classifier.fit(X_train, train_classes)
    class_predictions = classifier.predict(X_test)
    class_probabilities = classifier.predict_proba(X_test)[:, 1]
    metrics = {
        "mae": float(mean_absolute_error(y_test, regression_predictions)),
        "r2": float(r2_score(y_test, regression_predictions)),
        "accuracy": float(accuracy_score(test_classes, class_predictions)),
        "auc": float(roc_auc_score(test_classes, class_probabilities))
        if test_classes.nunique() == 2
        else 0.5,
        "train_through_year": float(cutoff_year - 1),
        "test_from_year": float(cutoff_year),
    }
    regression.fit(data[feature_names], data["NEXT_GDP_GROWTH"])
    classifier.fit(data[feature_names], (data["NEXT_GDP_GROWTH"] >= target_median).astype(int))
    return regression, classifier, metrics, target_median


@lru_cache(maxsize=1)
def get_bundle() -> dict[str, Any]:
    raw_data = load_world_bank_data()
    data, feature_names, preprocessing = prepare_dataset(raw_data)
    regression, classifier, metrics, target_median = train_models(data, feature_names)
    return {
        "data": data,
        "features": feature_names,
        "preprocessing": preprocessing,
        "regression": regression,
        "classifier": classifier,
        "metrics": metrics,
        "target_median": target_median,
        "record_ids": data["ID"].astype(int).tolist(),
    }


def create_chart(data: pd.DataFrame, feature_names: list[str], chart_type: str) -> BytesIO:
    from io import BytesIO

    figure, axis = plt.subplots(figsize=(8, 3.2), dpi=150)
    figure.patch.set_facecolor("#f8f8f4")
    axis.set_facecolor("#f8f8f4")
    if chart_type == "line":
        yearly = data.groupby("year")["NEXT_GDP_GROWTH"].mean()
        axis.plot(yearly.index, yearly.values, color="#176b5b", marker="o", linewidth=2.5)
        axis.set(xlabel="Year", ylabel="Average GDP growth (%)", title="Next-year growth by observation year")
    elif chart_type == "bar":
        latest_year = int(data["year"].max())
        latest = data[data["year"] == latest_year]
        regional = latest.groupby("region")["RENEWABLE_SHARE"].mean().dropna().sort_values()
        axis.barh(regional.index, regional.values, color="#176b5b")
        axis.set(xlabel="Renewable energy share (%)", title=f"Renewables by region · {latest_year}")
    elif chart_type == "histogram":
        axis.hist(data["NEXT_GDP_GROWTH"].dropna(), bins=22, color="#df7c5c", edgecolor="#f8f8f4")
        axis.set(xlabel="Next-year GDP growth (%)", ylabel="Country-year records", title="Growth forecast target distribution")
    elif chart_type == "scatter":
        plot_data = data.dropna(subset=["RENEWABLE_SHARE", "NEXT_GDP_GROWTH"])
        axis.scatter(
            plot_data["RENEWABLE_SHARE"], plot_data["NEXT_GDP_GROWTH"],
            s=18, alpha=0.52, color="#df7c5c", edgecolors="none",
        )
        axis.set(xlabel="Renewable energy share (%)", ylabel="Next-year GDP growth (%)", title="Energy transition and growth")
    elif chart_type == "heatmap":
        correlation = data[feature_names + ["NEXT_GDP_GROWTH"]].corr()
        labels = [FEATURE_LABELS.get(name, name) for name in correlation.columns]
        image = axis.imshow(correlation, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(range(len(labels)), labels=labels, rotation=42, ha="right", fontsize=7)
        axis.set_yticks(range(len(labels)), labels=labels, fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title("Correlation across development indicators", fontsize=10)
    else:
        plt.close(figure)
        raise ValueError("Unknown chart type")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x" if chart_type == "bar" else "y", alpha=0.16)
    figure.tight_layout()
    output = BytesIO()
    figure.savefig(output, format="png", facecolor=figure.get_facecolor())
    plt.close(figure)
    output.seek(0)
    return output
