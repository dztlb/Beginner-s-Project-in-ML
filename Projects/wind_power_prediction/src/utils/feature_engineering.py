# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import warnings

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor

def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = df[datetime_col]
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dow"] = dt.dt.dayofweek
    return df

def create_lag_features(df: pd.DataFrame, target_col: str, lags=[1,2,3,6,12]):
    df = df.copy()
    for l in lags:
        df[f"{target_col}_lag_{l}"] = df[target_col].shift(l)
    df = df.fillna(method="bfill")
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str, windows=[3,6,12]):
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"]  = df[target_col].rolling(w).std()
        df[f"{target_col}_roll_max_{w}"]  = df[target_col].rolling(w).max()
        df[f"{target_col}_roll_min_{w}"]  = df[target_col].rolling(w).min()
    df = df.fillna(method="bfill")
    return df

def create_cwt_features(df: pd.DataFrame, target_col: str, scales=None, wavelet="morl"):
    df = df.copy()
    if scales is None:
        scales = np.arange(1, 32)
    try:
        import pywt
        signal = df[target_col].astype(float).values
        coeffs, freqs = pywt.cwt(signal, scales, wavelet)
        # Aggregate statistics across scales per time step
        df[f"{target_col}_cwt_mean"] = coeffs.mean(axis=0)
        df[f"{target_col}_cwt_std"]  = coeffs.std(axis=0)
        df[f"{target_col}_cwt_max"]  = coeffs.max(axis=0)
        df[f"{target_col}_cwt_min"]  = coeffs.min(axis=0)
    except Exception as e:
        warnings.warn(f"CWT skipped: {e}")
    return df

def select_features_rfecv(X: pd.DataFrame, y: np.ndarray, cv=3):
    try:
        est = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rfecv = RFECV(estimator=est, step=1, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        rfecv.fit(X, y)
        mask = rfecv.support_
        selected = X.columns[mask].tolist()
        if not selected:
            selected = X.columns.tolist()
        print(f"[RFECV] {X.shape[1]} → {len(selected)} features")
        return selected, {"ranking": rfecv.ranking_.tolist()}
    except Exception as e:
        print("[RFECV] failed, using all features:", e)
        return X.columns.tolist(), {"error": str(e)}
