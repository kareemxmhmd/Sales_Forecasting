import pandas as pd
import numpy as np

class RecursiveForecaster:
    def __init__(self, model, preprocessor, features_manifest, quantile_models=None):
        self.model = model
        self.preprocessor = preprocessor
        self.features_manifest = features_manifest
        self.quantile_models = quantile_models
        
    def forecast(self, future_df: pd.DataFrame, historical_sales: dict):
        """
        historical_sales: dict mapping store_id -> list of last 14 days sales (newest last)
        future_df: DataFrame containing future rows to predict.
        """
        df = future_df.copy()
        df['_orig_index'] = df.index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(['Date', 'Store']).reset_index(drop=True)
        else:
            df = df.sort_values(['Store']).reset_index(drop=True)
            
        state = {int(store): list(history[-14:]) for store, history in historical_sales.items()}
        features = self.features_manifest['features']
        
        df['Sales_Pred'] = 0.0
        if self.quantile_models:
            df['Sales_P10'] = 0.0
            df['Sales_P90'] = 0.0
            
        group_col = 'Date' if 'Date' in df.columns else 'Store'
        
        for date_or_store, group in df.groupby(group_col):
            idx = group.index
            
            for i in idx:
                store = int(df.at[i, 'Store'])
                history = state.get(store, [0.0]*14)
                
                for lag in range(1, 15):
                    val = history[-lag] if len(history) >= lag else 0.0
                    df.at[i, f'Sales_lag_{lag}'] = val
                    
                recent_7 = history[-7:] if len(history) >= 7 else history
                df.at[i, 'Sales_rolling_mean_7'] = np.mean(recent_7) if len(recent_7) > 0 else 0.0
                df.at[i, 'Sales_rolling_std_7'] = np.std(recent_7, ddof=1) if len(recent_7) > 1 else 0.0
                
            day_df = df.loc[idx]
            
            # Prevent missing columns error if features are missing
            for f in features:
                if f not in day_df.columns:
                    day_df[f] = 0.0
                    
            X = self.preprocessor.transform(day_df[features])
            raw_preds = self.model.predict(X)
            
            if hasattr(raw_preds, 'flatten'):
                raw_preds = raw_preds.flatten()
                
            raw_preds = np.clip(raw_preds, -20.0, 20.0)
            preds = np.clip(np.expm1(raw_preds), 0, 1e7)
            
            if 'Open' in day_df.columns:
                open_mask = day_df['Open'].values != 0
                preds[~open_mask] = 0.0
            
            df.loc[idx, 'Sales_Pred'] = preds
            
            if self.quantile_models and "q10" in self.quantile_models and "q90" in self.quantile_models:
                p10_raw = self.quantile_models['q10'].predict(X)
                p90_raw = self.quantile_models['q90'].predict(X)
                p10_raw = np.clip(p10_raw, -20.0, 20.0)
                p90_raw = np.clip(p90_raw, -20.0, 20.0)
                p10 = np.clip(np.expm1(p10_raw), 0, 1e7)
                p90 = np.clip(np.expm1(p90_raw), 0, 1e7)
                if 'Open' in day_df.columns:
                    p10[~open_mask] = 0.0
                    p90[~open_mask] = 0.0
                df.loc[idx, 'Sales_P10'] = p10
                df.loc[idx, 'Sales_P90'] = p90
                
            # Update state
            for i, p in zip(idx, preds):
                store = int(df.at[i, 'Store'])
                if store not in state:
                    state[store] = [0.0]*14
                state[store].append(p)
                state[store] = state[store][-14:]
                
        # Restore original order
        df = df.sort_values('_orig_index').drop(columns=['_orig_index'])
        return df
