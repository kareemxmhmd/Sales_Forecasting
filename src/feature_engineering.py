import pandas as pd
from config import LOOKBACK_WINDOW


def extract_time_features(df: pd.DataFrame):
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

def create_lag_features(df: pd.DataFrame, target: str = 'Sales', lookback: int = LOOKBACK_WINDOW):
    df = df.copy()
    for lag in range(1, lookback + 1):
        df[f'{target}_lag_{lag}'] = df.groupby('Store')[target].shift(lag)
        
    df[f'{target}_rolling_mean_7'] = df.groupby('Store')[f'{target}_lag_1'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df[f'{target}_rolling_std_7'] = df.groupby('Store')[f'{target}_lag_1'].transform(lambda x: x.rolling(7, min_periods=1).std())
    
    return df

if __name__ == "__main__":
    from data_loader import load_data, merge_store_data
    from preprocessing import preprocess_basic
    
    train_df, test_df, store_df = load_data()
    merged_df = merge_store_data(train_df, store_df)
    preprocessed_df = preprocess_basic(merged_df)
    fe_df = extract_time_features(preprocessed_df)
    fe_df = create_lag_features(fe_df)
    print("Feature engineered data shape:", fe_df.shape)
