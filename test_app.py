import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import app
from sustainability import INDICATORS, prepare_dataset, train_models


def make_test_bundle():
    rows = []
    for country_number in range(8):
        country_code = f"TST{country_number}"
        for year in range(2000, 2020):
            rows.append(
                {
                    "country_code": country_code,
                    "country": f"Test Country {country_number}",
                    "region": f"Test Region {country_number % 2}",
                    "year": year,
                    "GDP_GROWTH": ((year + country_number) % 9) - 4.0,
                    "GDP_PER_CAPITA": 1000 + country_number * 500 + (year - 2000) * 100,
                    "RENEWABLE_SHARE": (country_number * 11 + year - 2000) % 90 + 5.0,
                    "ELECTRICITY_ACCESS": min(60 + country_number * 5 + year - 2000, 100.0),
                    "LIFE_EXPECTANCY": 55 + country_number * 2 + (year - 2000) * 0.4,
                    "FOREST_SHARE": (country_number * 9 + year - 2000) % 75 + 10.0,
                    "NEXT_GDP_GROWTH": ((year + country_number + 1) % 9) - 4.0,
                }
            )
    raw_data = pd.DataFrame(rows)
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


class SustainabilityApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = make_test_bundle()

    def setUp(self):
        self.bundle_patch = patch("app.get_bundle", return_value=self.bundle)
        self.bundle_patch.start()
        self.addCleanup(self.bundle_patch.stop)
        self.client = app.app.test_client()

    def test_overview_has_country_indicators_and_growth_metrics(self):
        response = self.client.get("/api/overview")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["records"], 160)
        self.assertEqual(payload["countries"], 8)
        self.assertIn("RENEWABLE_SHARE", payload["features"])
        self.assertIn("NEXT_GDP_GROWTH", payload["summary"]["mean"])
        self.assertEqual(payload["numpy_lab"]["reshape"], [4, 3, 3])
        self.assertIn("r2", payload["metrics"])
        self.assertEqual(sum(group["records"] for group in payload["groups"]), 160)

    def test_filter_sort_region_and_paginate_country_years(self):
        response = self.client.get(
            "/api/records?feature=RENEWABLE_SHARE&direction=asc&page_size=5&min=20&region=Test+Region+0"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        values = [row["RENEWABLE_SHARE"] for row in payload["records"]]
        self.assertEqual(len(values), 5)
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(value >= 20 for value in values))
        self.assertTrue(all(row["region"] == "Test Region 0" for row in payload["records"]))

    def test_dsa_merge_sort_and_binary_lookup(self):
        self.assertEqual(app.merge_sort_records([{"v": 3}, {"v": 1}, {"v": 2}], "v"), [{"v": 1}, {"v": 2}, {"v": 3}])
        self.assertEqual(app.binary_search_ids([1, 3, 5, 7], 5), 2)
        self.assertEqual(app.binary_search_ids([1, 3, 5, 7], 4), -1)
        self.assertEqual(self.client.get("/api/record/160").status_code, 200)
        self.assertEqual(self.client.get("/api/record/999").status_code, 404)

    def test_prediction_estimates_growth_and_growth_group(self):
        row = self.bundle["data"].iloc[-1]
        payload = {name.lower(): float(row[name]) for name in self.bundle["features"]}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("forecast_growth", response.json)
        self.assertIn(response.json["predicted_group"], {"Above typical", "Below typical"})

    def test_country_series_returns_ordered_years(self):
        response = self.client.get("/api/series/TST0")
        self.assertEqual(response.status_code, 200)
        years = [row["year"] for row in response.json["records"]]
        self.assertEqual(years, sorted(years))

    def test_matplotlib_chart_routes(self):
        for chart_type in ("line", "bar", "histogram", "scatter", "heatmap"):
            with self.subTest(chart=chart_type):
                response = self.client.get(f"/api/chart/{chart_type}.png")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                self.assertTrue(response.data.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
