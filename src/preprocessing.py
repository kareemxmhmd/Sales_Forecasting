import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from config import PREPROCESSOR_PATH, FEATURE_MANIFEST_PATH
import json

def preprocess_basic(df: pd.DataFrame):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(0)
    return df

def clean_data_for_training(df: pd.DataFrame):
    if 'Sales' in df.columns:
        df = df[(df['Open'] != 0) & (df['Sales'] > 0)]
    df = df.dropna().reset_index(drop=True)
    return df

def build_and_save_preprocessor(train_df: pd.DataFrame, categorical_cols: list, numerical_cols: list):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    preprocessor.fit(train_df)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    features = categorical_cols + numerical_cols
    with open(FEATURE_MANIFEST_PATH, 'w') as f:
        json.dump({"features": features, "categorical": categorical_cols, "numerical": numerical_cols}, f)
        
    return preprocessor

def apply_preprocessor(df: pd.DataFrame):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    with open(FEATURE_MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)
        
    features = manifest['features']
    return preprocessor.transform(df[features])

if __name__ == "__main__":
    from data_loader import load_data, merge_store_data

    train_df, test_df, store_df = load_data()
    merged_df = merge_store_data(train_df, store_df)
    processed_df = preprocess_basic(merged_df)
    print("Preprocessed data shape:", processed_df.shape)
