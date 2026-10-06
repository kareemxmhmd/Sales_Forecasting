import sys
from pathlib import Path

# Add repository root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_healthy():
    """Health check must strictly return HTTP 200 with artifacts loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["artifacts_loaded"] is True
    assert "model_name" in payload
    assert payload["model_type"] == "sklearn"

def test_get_metadata():
    """Metadata endpoint returns valid training information and metrics."""
    response = client.get("/metadata")
    assert response.status_code == 200
    meta = response.json()
    assert "best_model" in meta
    assert meta["best_model"] == "LGBMRegressor"
    assert "all_metrics" in meta
    assert "feature_columns" in meta
    assert len(meta["feature_columns"]) == 35

def test_predict_validation_errors():
    """Empty or malformed bodies must return validation errors."""
    resp_empty = client.post("/predict", json={})
    assert resp_empty.status_code == 422

    resp_empty_data = client.post("/predict", json={"data": []})
    assert resp_empty_data.status_code == 400

def test_predict_open_store_positive_sales():
    """Open store should yield positive sales within expected range."""
    payload = {
        "data": [
            {
                "Store": 1,
                "DayOfWeek": 5,
                "Date": "2015-07-31",
                "Open": 1,
                "Promo": 1,
                "StateHoliday": "0",
                "SchoolHoliday": 1,
                "StoreType": "c",
                "Assortment": "a",
                "CompetitionDistance": 1270.0,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] > 0.0
    assert "results" in body
    res = body["results"][0]
    assert res["store"] == 1
    assert res["open"] == 1
    assert res["prediction"] > 0.0
    assert res["prediction_interval_p10"] is not None
    assert res["prediction_interval_p90"] is not None
    assert res["prediction_interval_p10"] <= res["prediction_interval_p90"]

def test_predict_closed_store_zero_sales():
    """Closed stores must strictly yield 0.0 sales."""
    payload = {
        "data": [
            {
                "Store": 1,
                "DayOfWeek": 7,
                "Date": "2015-08-02",
                "Open": 0,
                "Promo": 0,
                "StateHoliday": "0",
                "SchoolHoliday": 0,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"][0] == 0.0
    res = body["results"][0]
    assert res["prediction"] == 0.0
    assert res["prediction_interval_p10"] == 0.0
    assert res["prediction_interval_p90"] == 0.0

def test_predict_batch_mixed_stores():
    """Batch predictions should handle mixed open/closed stores correctly."""
    payload = {
        "data": [
            {"Store": 1, "DayOfWeek": 5, "Date": "2015-07-31", "Open": 1, "Promo": 1},
            {"Store": 2, "DayOfWeek": 7, "Date": "2015-08-02", "Open": 0, "Promo": 0},
            {"Store": 3, "DayOfWeek": 2, "Date": "2015-07-28", "Open": 1, "Promo": 0},
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 3
    assert body["predictions"][0] > 0.0
    assert body["predictions"][1] == 0.0
    assert body["predictions"][2] > 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
