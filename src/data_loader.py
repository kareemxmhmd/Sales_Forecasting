import pandas as pd
try:
    from config import TRAIN_DATA_PATH, TEST_DATA_PATH, STORE_DATA_PATH
except ImportError:
    from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, STORE_DATA_PATH

def load_data(
    train_path: str = TRAIN_DATA_PATH, 
    test_path: str = TEST_DATA_PATH, 
    store_path: str = STORE_DATA_PATH
):
    train_df = pd.read_csv(train_path, low_memory=False)
    print("Train data shape:", train_df.shape)
    
    test_df = pd.read_csv(test_path, low_memory=False)
    print("Test data shape:", test_df.shape)
    
    store_df = pd.read_csv(store_path, low_memory=False)
    print("Store data shape:", store_df.shape)
    return train_df, test_df, store_df

def merge_store_data(data_df: pd.DataFrame, store_df: pd.DataFrame):
    merged_df = pd.merge(data_df, store_df, on="Store", how="left")
    print("Merged data shape:", merged_df.shape)
    return merged_df

if __name__ == "__main__":
    df = load_data()
    merged_store_df = merge_store_data(df[0], df[2])
