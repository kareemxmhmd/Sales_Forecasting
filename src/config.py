from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TRAIN_DATA_PATH = DATA_DIR / "raw" / "train.csv"
TEST_DATA_PATH = DATA_DIR / "raw" / "test.csv"
STORE_DATA_PATH = DATA_DIR / "raw" / "store.csv"

MODEL_PATH = MODELS_DIR / "sales_model.keras"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
FEATURE_MANIFEST_PATH = ARTIFACTS_DIR / "features.json"

BEST_MODEL_PATH = MODELS_DIR / "model.pkl"
NN_MODEL_PATH = MODELS_DIR / "nn_model.keras"
QUANTILE_MODELS_PATH = MODELS_DIR / "quantile_models.pkl"
METRICS_ML_PATH = MODELS_DIR / "metrics_ml.json"
METRICS_NN_PATH = MODELS_DIR / "metrics_nn.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
NN_ENCODERS_PATH = MODELS_DIR / "nn_encoders.json"
FEATURE_COLUMNS_ML_PATH = MODELS_DIR / "feature_columns_ml.json"
FEATURE_COLUMNS_NN_PATH = MODELS_DIR / "feature_columns_nn.json"
HORIZON_DAYS = 42
HOLDOUT_DAYS = 14

LOOKBACK_WINDOW = 14
TARGET = "Sales"