import json
from pathlib import Path
from data_loader import load_data, merge_store_data
from preprocessing import preprocess_basic
from config import ARTIFACTS_DIR

def build_cache():
    print("Loading raw dataset to build historical sales cache...")
    train_df, _, store_df = load_data()
    df = merge_store_data(train_df, store_df)
    
    print("Preprocessing to sort chronologically...")
    df = preprocess_basic(df)
    
    cache = {}
    grouped = df.groupby('Store')
    for store, group in grouped:
        sales = group['Sales'].fillna(0).tail(14).tolist()
        cache[int(store)] = sales
        
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "historical_sales.json"
    
    with open(out_path, 'w') as f:
        json.dump(cache, f)
        
    print(f"Saved historical sales cache for {len(cache)} stores to {out_path}")

if __name__ == "__main__":
    build_cache()
