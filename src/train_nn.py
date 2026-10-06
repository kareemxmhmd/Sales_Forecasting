import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    NN_MODEL_PATH, METRICS_NN_PATH, FEATURE_COLUMNS_NN_PATH,
    NN_ENCODERS_PATH, TARGET
)
from training_utils import (
    get_dataset, get_columns, time_split, tail_holdout,
    leakage_report, metrics, save_json, get_models_dir
)
from preprocessing import build_and_save_preprocessor, apply_preprocessor

def build_mlp_model(input_dim):
    inp = Input(shape=(input_dim,), name="mlp_input")
    x = Dense(128, activation="relu")(inp)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inputs=inp, outputs=out, name="nn_mlp")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

def build_embedding_model(cat_dims, num_dim):
    cat_inputs = []
    cat_embeddings = []

    for col, (num_classes, emb_dim) in cat_dims.items():
        inp = Input(shape=(1,), name=f"input_{col}")
        emb = Embedding(input_dim=num_classes + 1, output_dim=emb_dim, name=f"emb_{col}")(inp)
        flat = Flatten(name=f"flat_{col}")(emb)
        cat_inputs.append(inp)
        cat_embeddings.append(flat)

    num_input = Input(shape=(num_dim,), name="input_numerical")
    concat = Concatenate(name="concat_all")(cat_embeddings + [num_input])
    x = Dense(128, activation="relu")(concat)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="linear")(x)

    model = Model(inputs=cat_inputs + [num_input], outputs=out, name="nn_embedding")
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

def fit_label_encoders(df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        unique_vals = df[col].astype(str).dropna().unique().tolist()
        encoders[col] = {str(val): idx + 1 for idx, val in enumerate(unique_vals)}
    return encoders

def transform_categorical(df, encoders, categorical_cols):
    encoded = {}
    for col in categorical_cols:
        mapping = encoders[col]
        encoded[col] = df[col].astype(str).map(mapping).fillna(0).astype(int).values
    return encoded

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    get_models_dir()

    print("Loading dataset for neural network training...")
    df = get_dataset()

    print("Defining feature columns...")
    categorical_cols, numerical_cols = get_columns(df)
    feature_cols = numerical_cols + categorical_cols

    print("Performing 42-day time split...")
    train_df, val_df = time_split(df)

    print("Performing 14-day tail holdout split for early stopping...")
    sub_train, sub_val = tail_holdout(train_df, days=14)

    print("Running leakage check...")
    leakage_report(train_df, val_df, feature_cols)

    y_sub_train_log = np.log1p(sub_train[TARGET].values)
    y_sub_val_log = np.log1p(sub_val[TARGET].values)
    y_val_true = val_df[TARGET].values

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print("\nCandidate 1: Training nn_mlp (apply_preprocessor)")
    build_and_save_preprocessor(train_df, categorical_cols, numerical_cols)
    X_sub_train = apply_preprocessor(sub_train)
    X_sub_val = apply_preprocessor(sub_val)
    X_val = apply_preprocessor(val_df)

    mlp_model = build_mlp_model(X_sub_train.shape[1])
    mlp_model.fit(
        X_sub_train, y_sub_train_log,
        validation_data=(X_sub_val, y_sub_val_log),
        epochs=50, batch_size=256,
        callbacks=callbacks, verbose=1
    )

    mlp_preds_log = mlp_model.predict(X_val, batch_size=1024).flatten()
    mlp_preds = np.clip(np.expm1(mlp_preds_log), 0, None)
    mlp_metrics = metrics(y_val_true, mlp_preds)
    print(f"nn_mlp -> MAE: {mlp_metrics['mae']:.2f}, RMSE: {mlp_metrics['rmse']:.2f}, RMSPE: {mlp_metrics['rmspe']:.4f}")

    print("\nCandidate 2: Training nn_embedding (Entity Embeddings) ---")
    nn_encoders = fit_label_encoders(train_df, categorical_cols)
    save_json(nn_encoders, NN_ENCODERS_PATH)

    emb_dims_spec = {
        "Store": 10,
        "DayOfWeek": 4,
        "StoreType": 2,
        "Assortment": 2,
        "StateHoliday": 2,
        "SchoolHoliday": 2,
        "PromoInterval": 2
    }
    cat_dims = {col: (len(nn_encoders[col]), emb_dims_spec.get(col, 2)) for col in categorical_cols}

    scaler = StandardScaler()
    num_sub_train = scaler.fit_transform(sub_train[numerical_cols])
    num_sub_val = scaler.transform(sub_val[numerical_cols])
    num_val = scaler.transform(val_df[numerical_cols])

    sub_train_cat = transform_categorical(sub_train, nn_encoders, categorical_cols)
    sub_val_cat = transform_categorical(sub_val, nn_encoders, categorical_cols)
    val_cat = transform_categorical(val_df, nn_encoders, categorical_cols)

    inputs_train = [sub_train_cat[c] for c in categorical_cols] + [num_sub_train]
    inputs_val_holdout = [sub_val_cat[c] for c in categorical_cols] + [num_sub_val]
    inputs_val_test = [val_cat[c] for c in categorical_cols] + [num_val]

    emb_model = build_embedding_model(cat_dims, len(numerical_cols))
    emb_model.fit(
        inputs_train, y_sub_train_log,
        validation_data=(inputs_val_holdout, y_sub_val_log),
        epochs=50, batch_size=256,
        callbacks=callbacks, verbose=1
    )

    emb_preds_log = emb_model.predict(inputs_val_test, batch_size=1024).flatten()
    emb_preds = np.clip(np.expm1(emb_preds_log), 0, None)
    emb_metrics = metrics(y_val_true, emb_preds)
    print(f"nn_embedding -> MAE: {emb_metrics['mae']:.2f}, RMSE: {emb_metrics['rmse']:.2f}, RMSPE: {emb_metrics['rmspe']:.4f}")

    all_nn_metrics = {
        "nn_mlp": mlp_metrics,
        "nn_embedding": emb_metrics
    }
    leakage_report(train_df, val_df, feature_cols, model_metrics=all_nn_metrics)

    best_variant = "nn_mlp" if mlp_metrics["rmspe"] <= emb_metrics["rmspe"] else "nn_embedding"
    best_nn_model = mlp_model if best_variant == "nn_mlp" else emb_model
    best_metrics = all_nn_metrics[best_variant]

    print(f"\nBest NN variant selected: {best_variant} (RMSPE: {best_metrics['rmspe']:.4f})")

    print(f"Saving best NN model to {NN_MODEL_PATH}...")
    best_nn_model.save(NN_MODEL_PATH)

    metrics_payload = {
        "best_variant": best_variant,
        "metrics": best_metrics,
        "all": all_nn_metrics
    }
    save_json(metrics_payload, METRICS_NN_PATH)
    save_json(feature_cols, FEATURE_COLUMNS_NN_PATH)
    print("Neural network training complete!")

if __name__ == "__main__":
    main()
