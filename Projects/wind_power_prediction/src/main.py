# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.data_processing import (
    basic_inspect, ensure_datetime, kpss_check,
    sliding_window_impute, detect_and_correct_outliers
)
from src.utils.feature_engineering import (
    create_time_features, create_lag_features, create_rolling_features,
    create_cwt_features, select_features_rfecv
)
from src.utils.model_utils import create_sequences, save_json, seed_everything
from src.models.random_forest import train_random_forest_asso
from src.models.deep_learning import build_tcn_attention_bigru, train_dl_model

def run_pipeline(csv_path, datetime_col="DATETIME", target_col="POWER",
                 save_prefix="run1", time_steps=10, test_size=0.2, random_state=42):

    seed_everything(42)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # 1) Load
    df = pd.read_csv(csv_path)
    basic_inspect(df)

    # 2) Datetime & sort
    df = ensure_datetime(df, datetime_col)
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # 3) KPSS
    kpss_res = kpss_check(df[target_col].dropna().values)
    save_json(kpss_res, f"results/logs/{save_prefix}_kpss.json")

    # 4) Impute missing
    for col in df.columns:
        if df[col].isna().sum() > 0:
            df[col] = sliding_window_impute(df[col], window_size=5)

    # 5) Outliers detect & correct using features except datetime/target
    base_features = [c for c in df.columns if c not in [datetime_col, target_col]]
    if base_features:
        df[target_col] = detect_and_correct_outliers(
            df, target_col=target_col, feature_cols=base_features
        )

    # 6) Feature engineering
    df = create_time_features(df, datetime_col)
    df = create_lag_features(df, target_col, lags=[1,2,3,6,12,24])
    df = create_rolling_features(df, target_col, windows=[3,6,12,24])
    df = create_cwt_features(df, target_col)  # safe if pywt present

    # 7) Prepare matrices
    feature_cols = [c for c in df.columns if c not in [datetime_col, target_col]]
    X = df[feature_cols].fillna(0.0)
    y = df[target_col].values

    # 8) RFECV selection
    selected_features, rfecv_info = select_features_rfecv(X, y)
    save_json({"selected_features": selected_features}, f"results/logs/{save_prefix}_selected_features.json")
    X_selected = X[selected_features].values if selected_features else X.values

    # 9) Split & scale (time-ordered split)
    split_idx = int(len(X_selected)*(1.0-test_size))
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 10a) RandomForest + ASSO
    rf_results = train_random_forest_asso(X_train, y_train, X_test, y_test, save_prefix=save_prefix)

    # 10b) DL model (optional if tensorflow available)
    dl_metrics = None
    try:
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps=time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps=time_steps)
        model = build_tcn_attention_bigru(input_shape=(time_steps, X_train.shape[1]))
        history, preds, dl_metrics = train_dl_model(model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, save_prefix=save_prefix)
    except Exception as e:
        dl_metrics = {"note": "Deep learning step skipped (likely TensorFlow not installed).", "error": str(e)}

    # 11) Compare
    from pprint import pprint
    summary = {
        "rf": rf_results["metrics_test"],
        "dl": dl_metrics
    }
    pprint(summary)
    save_json(summary, f"results/logs/{save_prefix}_summary.json")

    print("\nPipeline finished. Artifacts saved under results/.")
