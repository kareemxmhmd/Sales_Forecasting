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

LOOKBACK_WINDOW = 14
TARGET = "Sales"
