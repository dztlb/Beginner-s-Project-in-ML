# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def basic_inspect(df: pd.DataFrame):
    print(f"[INFO] shape={df.shape}")
    print(f"[INFO] columns={list(df.columns)}")
    print(df.head())

def ensure_datetime(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    if df[datetime_col].isna().any():
        raise ValueError(f"Datetime parsing failed for column: {datetime_col}")
    return df

def kpss_check(series_values: np.ndarray):
    try:
        stat, pval, lags, crit = kpss(series_values, regression='c', nlags="auto")
        res = {"KPSS_stat": float(stat), "p_value": float(pval), "lags": int(lags), "critical_values": crit}
        res["stationary?"] = pval >= 0.05
        print(f"[KPSS] stat={stat:.4f} p={pval:.4f} stationary={res['stationary?']}")
        return res
    except Exception as e:
        print("[KPSS] failed:", e)
        return {"error": str(e)}

def sliding_window_impute(series: pd.Series, window_size=5):
    s = series.copy()
    idxs = np.where(s.isna())[0]
    for i in idxs:
        lo = max(0, i-window_size)
        hi = min(len(s), i+window_size+1)
        val = s.iloc[lo:hi].dropna().mean()
        if np.isnan(val):
            val = s.mean()
        s.iloc[i] = val
    return s

def detect_and_correct_outliers(df, target_col, feature_cols):
    X = df[feature_cols].astype(float).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.1, random_state=42)
    labels = iso.fit_predict(Xs)
    out_idx = np.where(labels == -1)[0]
    in_idx = np.where(labels == 1)[0]
    print(f"[Outliers] detected: {len(out_idx)}")

    y = df[target_col].astype(float).values
    if len(out_idx) > 0 and len(in_idx) > 10:
        svr = SVR(kernel="rbf")
        svr.fit(Xs[in_idx], y[in_idx])
        y_corr = y.copy()
        y_corr[out_idx] = svr.predict(Xs[out_idx])
        return y_corr
    return y
