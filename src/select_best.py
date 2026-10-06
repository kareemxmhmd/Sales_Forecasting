import sys
from pathlib import Path
import json
from config import (
    METRICS_ML_PATH, METRICS_NN_PATH, METADATA_PATH,
    FEATURE_COLUMNS_ML_PATH, FEATURE_COLUMNS_NN_PATH
)

def save_json(data, filepath):
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    if not METRICS_ML_PATH.exists():
        print(f"Error: {METRICS_ML_PATH} not found. Run train_model.py first.")
        sys.exit(1)

    if not METRICS_NN_PATH.exists():
        print(f"Error: {METRICS_NN_PATH} not found. Run train_nn.py first.")
        sys.exit(1)

    with open(METRICS_ML_PATH, "r") as f:
        ml_data = json.load(f)

    with open(METRICS_NN_PATH, "r") as f:
        nn_data = json.load(f)

    table_rows = []

    if "baseline" in ml_data:
        b = ml_data["baseline"]
        table_rows.append({"model": "SeasonalNaive_Lag7", "mae": b["mae"], "rmse": b["rmse"], "rmspe": b["rmspe"], "type": "baseline"})

    for name, m in ml_data.get("all", {}).items():
        table_rows.append({"model": name, "mae": m["mae"], "rmse": m["rmse"], "rmspe": m["rmspe"], "type": "sklearn"})

    for name, m in nn_data.get("all", {}).items():
        table_rows.append({"model": name, "mae": m["mae"], "rmse": m["rmse"], "rmspe": m["rmspe"], "type": "keras"})

    table_rows.sort(key=lambda r: r["rmspe"])

    print("\n" + "=" * 68)
    print(f"{'Model':<25} {'MAE':<12} {'RMSE':<12} {'RMSPE':<12}")
    print("-" * 68)
    for r in table_rows:
        print(f"{r['model']:<25} {r['mae']:<12.2f} {r['rmse']:<12.2f} {r['rmspe']:<12.4f}")
    print("=" * 68)

    winner = table_rows[0]
    print(f"\nWinning Model: {winner['model']} (RMSPE: {winner['rmspe']:.4f})")

    is_keras = winner["type"] == "keras"
    model_type = "keras" if is_keras else "sklearn"
    model_file = "nn_model.keras" if is_keras else "model.pkl"
    feature_file = FEATURE_COLUMNS_NN_PATH if is_keras else FEATURE_COLUMNS_ML_PATH

    with open(feature_file, "r") as f:
        feature_cols = json.load(f)

    all_metrics = {r["model"]: {"mae": r["mae"], "rmse": r["rmse"], "rmspe": r["rmspe"]} for r in table_rows}

    metadata = {
        "best_model": winner["model"],
        "model_type": model_type,
        "model_file": model_file,
        "feature_columns": feature_cols,
        "target_transform": "log1p",
        "horizon_days": 42,
        "validation_days": 42,
        "all_metrics": all_metrics,
        "interval_metrics": ml_data.get("interval", {}),
        "training_date_range": ml_data.get("training_date_range", {"min": "unknown", "max": "unknown"})
    }

    save_json(metadata, METADATA_PATH)
    print(f"Metadata saved successfully to {METADATA_PATH}")

if __name__ == "__main__":
    main()
