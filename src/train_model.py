import numpy as np
import joblib
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import (
    BEST_MODEL_PATH, QUANTILE_MODELS_PATH, METRICS_ML_PATH,
    FEATURE_COLUMNS_ML_PATH, TARGET, PREPROCESSOR_PATH
)
from training_utils import (
    get_dataset, get_columns, time_split, leakage_report,
    metrics, save_json, get_models_dir
)
from preprocessing import build_and_save_preprocessor, apply_preprocessor

def train_and_eval_ml(X_train, y_train_log, val_df, train_df, preprocessor, feature_manifest):
    from recursive_forecast import RecursiveForecaster
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, random_state=42, n_jobs=-1),
        "LGBMRegressor": LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
    }

    # Extract historical sales from train_df to seed the recursive forecaster
    historical_sales = {}
    for store, group in train_df.groupby('Store'):
        historical_sales[int(store)] = group['Sales'].tail(14).tolist()

    all_metrics = {}
    fitted_models = {}

    y_val_true = val_df[TARGET].values

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_log)
        
        print(f"Evaluating {name} recursively over 42 days...")
        forecaster = RecursiveForecaster(model, preprocessor, feature_manifest)
        val_preds_df = forecaster.forecast(val_df, historical_sales)
        
        preds = val_preds_df['Sales_Pred'].values
        score = metrics(y_val_true, preds)
        print(f"  {name} -> MAE: {score['mae']:.2f}, RMSE: {score['rmse']:.2f}, RMSPE: {score['rmspe']:.4f}")
        all_metrics[name] = score
        fitted_models[name] = model

    best_name = min(all_metrics.keys(), key=lambda k: all_metrics[k]["rmspe"])
    print(f"Best ML model selected by lowest RMSPE: {best_name} ({all_metrics[best_name]['rmspe']:.4f})")
    return best_name, fitted_models[best_name], all_metrics

def train_quantiles(X_train, y_train_log, X_val, y_val_true):
    print("Training quantile regression models (alpha=0.1, alpha=0.9)...")
    q10 = LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
    q90 = LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)

    q10.fit(X_train, y_train_log)
    q90.fit(X_train, y_train_log)

    pred_q10 = np.clip(np.expm1(q10.predict(X_val)), 0, None)
    pred_q90 = np.clip(np.expm1(q90.predict(X_val)), 0, None)

    coverage = float(np.mean((y_val_true >= pred_q10) & (y_val_true <= pred_q90)))
    mean_width = float(np.mean(pred_q90 - pred_q10))

    print(f"Quantile interval coverage: {coverage * 100:.2f}% (nominal 80%)")
    print(f"Quantile interval mean width: {mean_width:.2f}")

    return {"q10": q10, "q90": q90}, {"coverage": round(coverage, 4), "mean_width": round(mean_width, 2)}

def main():
    np.random.seed(42)
    get_models_dir()

    print("Loading and preparing dataset...")
    df = get_dataset()

    print("Defining categorical and numerical feature sets...")
    categorical_cols, numerical_cols = get_columns(df)
    feature_cols = numerical_cols + categorical_cols
    print(f"Total features: {len(feature_cols)} ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")

    print("Chronological time split...")
    train_df, val_df = time_split(df)

    print("Running data leakage check...")
    leakage_report(train_df, val_df, feature_cols)

    print("Fitting preprocessor on train set and applying transformation...")
    print(f"Missing values in train before transform: {train_df[feature_cols].isna().sum().sum()}")
    print(f"Missing values in val before transform: {val_df[feature_cols].isna().sum().sum()}")
    build_and_save_preprocessor(train_df, categorical_cols, numerical_cols)
    X_train = apply_preprocessor(train_df)
    X_val = apply_preprocessor(val_df)
    print(f"Transformed X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    y_train_log = np.log1p(train_df[TARGET].values)
    y_val_true = val_df[TARGET].values

    print("Evaluating seasonal naive baseline (Sales_lag_7)...")
    valid_mask = val_df["Sales_lag_7"].notna()
    y_val_base_true = val_df.loc[valid_mask, TARGET].values
    y_val_base_pred = val_df.loc[valid_mask, "Sales_lag_7"].values
    baseline_metrics = metrics(y_val_base_true, y_val_base_pred)
    print(f"Baseline -> MAE: {baseline_metrics['mae']:.2f}, RMSE: {baseline_metrics['rmse']:.2f}, RMSPE: {baseline_metrics['rmspe']:.4f}")

    print("Training ML models...")
    # Load manifest explicitly since it was saved by build_and_save_preprocessor
    import json
    from config import FEATURE_MANIFEST_PATH
    with open(FEATURE_MANIFEST_PATH, 'r') as f:
        feature_manifest = json.load(f)
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    best_name, best_model, all_metrics = train_and_eval_ml(X_train, y_train_log, val_df, train_df, preprocessor, feature_manifest)

    print("Checking RMSPE leakage threshold across models...")
    leakage_report(train_df, val_df, feature_cols, model_metrics=all_metrics)

    print("Training prediction intervals...")
    quantile_models, interval_metrics = train_quantiles(X_train, y_train_log, X_val, y_val_true)

    print("Saving models and metrics artifacts...")
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Saved best model to {BEST_MODEL_PATH}")

    joblib.dump(quantile_models, QUANTILE_MODELS_PATH)
    print(f"Saved quantile models to {QUANTILE_MODELS_PATH}")

    ml_metrics_payload = {
        "best_model": best_name,
        "all": all_metrics,
        "baseline": baseline_metrics,
        "interval": interval_metrics,
        "training_date_range": {
            "min": train_df["Date"].min().strftime("%Y-%m-%d"),
            "max": train_df["Date"].max().strftime("%Y-%m-%d")
        }
    }
    save_json(ml_metrics_payload, METRICS_ML_PATH)
    save_json(feature_cols, FEATURE_COLUMNS_ML_PATH)

    print("Classic ML training complete!")

if __name__ == "__main__":
    main()
