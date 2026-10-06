import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import TARGET, LOOKBACK_WINDOW, MODELS_DIR
from data_loader import load_data, merge_store_data
from preprocessing import preprocess_basic, clean_data_for_training
from feature_engineering import extract_time_features, create_lag_features
from evaluate import rmspe_np

def get_dataset():
    train_df, _, store_df = load_data()
    sample_stores = os.environ.get("SAMPLE_STORES")
    if sample_stores:
        n_stores = int(sample_stores)
        unique_stores = train_df["Store"].unique()[:n_stores]
        train_df = train_df[train_df["Store"].isin(unique_stores)].copy()
        store_df = store_df[store_df["Store"].isin(unique_stores)].copy()
        print(f"SAMPLE_STORES set: filtered to {n_stores} stores ({len(train_df)} raw rows).")

    df = merge_store_data(train_df, store_df)
    df = preprocess_basic(df)
    df = extract_time_features(df)
    df = create_lag_features(df, target=TARGET, lookback=LOOKBACK_WINDOW)
    df = clean_data_for_training(df)

    print(f"Row count before Open/Sales filter: {len(df)}")
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
    print(f"Row count after Open/Sales filter: {len(df)}")

    df = df.sort_values("Date").reset_index(drop=True)
    return df

def get_columns(df):
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str)

    categorical_cols = [
        "Store", "DayOfWeek", "StoreType", "Assortment",
        "StateHoliday", "SchoolHoliday", "PromoInterval"
    ]
    numerical_cols = [
        "Open", "Promo", "CompetitionDistance", "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
        "Year", "Month", "Day", "WeekOfYear",
        "Sales_rolling_mean_7", "Sales_rolling_std_7"
    ]
    for i in range(1, LOOKBACK_WINDOW + 1):
        numerical_cols.append(f"Sales_lag_{i}")

    all_features = categorical_cols + numerical_cols
    assert "Customers" not in all_features, "Leakage detected: Customers in feature list!"
    assert "Sales" not in all_features, "Leakage detected: Sales in feature list!"
    assert "Date" not in all_features, "Leakage detected: Date in feature list!"

    for col in all_features:
        assert col in df.columns, f"Required feature missing in dataframe: {col}"

    return categorical_cols, numerical_cols

def time_split(df):
    cutoff = df["Date"].max() - pd.Timedelta(days=42)
    train_df = df[df["Date"] <= cutoff].copy()
    val_df = df[df["Date"] > cutoff].copy()

    assert train_df["Date"].max() < val_df["Date"].min(), "Time split violation: train date overlaps with validation date!"

    print(f"Train split: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    print(f"Val split:   {val_df['Date'].min().strftime('%Y-%m-%d')} to {val_df['Date'].max().strftime('%Y-%m-%d')} ({len(val_df)} rows)")
    return train_df, val_df

def tail_holdout(train_df, days=14):
    cutoff = train_df["Date"].max() - pd.Timedelta(days=days)
    sub_train = train_df[train_df["Date"] <= cutoff].copy()
    sub_val = train_df[train_df["Date"] > cutoff].copy()

    assert sub_train["Date"].max() < sub_val["Date"].min(), "Tail holdout violation: sub_train date overlaps with sub_val date!"

    print(f"Sub-train:    {sub_train['Date'].min().strftime('%Y-%m-%d')} to {sub_train['Date'].max().strftime('%Y-%m-%d')} ({len(sub_train)} rows)")
    print(f"Tail holdout: {sub_val['Date'].min().strftime('%Y-%m-%d')} to {sub_val['Date'].max().strftime('%Y-%m-%d')} ({len(sub_val)} rows)")
    return sub_train, sub_val

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    rmspe_val = float(rmspe_np(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "rmspe": round(rmspe_val, 4)}

def leakage_report(train_df, val_df, feature_cols, model_metrics=None):
    lags = []
    for f in feature_cols:
        m = re.search(r"lag_(\d+)", f)
        if m:
            lags.append(int(m.group(1)))
    min_lag = min(lags) if lags else None
    print(f"[Leakage Check] Smallest lag in features: {min_lag}")

    rolling_features = [f for f in feature_cols if "rolling" in f]
    print(f"[Leakage Check] Rolling features detected: {rolling_features} (computed from Sales_lag_1, shift=1)")

    if min_lag is not None and min_lag < 42:
        print(f"WARNING: Validation scores are optimistic! Smallest lag is {min_lag} (< 42). In a true 42-day forecast horizon, lags < 42 would not be directly observed without recursive roll-forward.")

    overlap = pd.merge(train_df[["Store", "Date"]], val_df[["Store", "Date"]], on=["Store", "Date"])
    print(f"[Leakage Check] Duplicate (Store, Date) pairs between train and val: {len(overlap)}")

    if model_metrics:
        for name, m_dict in model_metrics.items():
            if isinstance(m_dict, dict) and "rmspe" in m_dict:
                score = m_dict["rmspe"]
                if score < 0.05:
                    print(f"WARNING: Model '{name}' RMSPE is {score:.4f} (< 0.05). Suspiciously good, check for leakage!")

def save_json(data, filepath):
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON artifact to {path}")

def get_models_dir():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
