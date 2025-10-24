from __future__ import annotations
import os, warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 工具：类别列统一为字符串（顶层函数可被pickle） ----------
def to_str_array(X):
    try:
        return X.astype(str)
    except Exception:
        return np.asarray(X).astype(str)

# ---------- 预处理：数值标准化 + 类别OneHot（先转字符串） ----------
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline([("scaler", StandardScaler())])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.4
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.4
    cat_tf = Pipeline([
        ("to_str", FunctionTransformer(to_str_array, feature_names_out="one-to-one")),
        ("oh", ohe),
    ])
    return ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])

# ---------- 三个模型与搜索空间：RF / XGB / STK(RF+XGB→LR) ----------
def model_spaces(seed: int, n_estimators_rf: int = 400, n_estimators_xgb: int = 400) -> Dict[str, Tuple[object, Dict]]:
    rf = RandomForestClassifier(random_state=seed, class_weight="balanced", n_estimators=n_estimators_rf)
    rf_space = {
        "clf__max_depth": [None, 10, 20],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }

    xgb = XGBClassifier(
        random_state=seed, eval_metric="mlogloss", tree_method="hist",
        n_estimators=n_estimators_xgb
    )
    xgb_space = {
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__min_child_weight": [1, 3],
        # 轻度正则可提升稳定性（可选）
        # "clf__reg_lambda": [1.0, 2.0],
    }

    # Stacking：折外概率 + 是否传原始特征由搜索决定
    cv_stk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    base_rf  = RandomForestClassifier(random_state=seed, class_weight="balanced", n_estimators=n_estimators_rf)
    base_xgb = XGBClassifier(random_state=seed, eval_metric="mlogloss", tree_method="hist", n_estimators=n_estimators_xgb)
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    stk = StackingClassifier(
        estimators=[("rf", base_rf), ("xgb", base_xgb)],
        final_estimator=meta,
        stack_method="predict_proba",
        passthrough=True,          # 交给搜索决定是否保留
        cv=cv_stk,
        n_jobs=None
    )
    stk_space = {
        "clf__final_estimator__C": [0.5, 1.0, 2.0],
        "clf__xgb__max_depth": [3, 5],
        "clf__xgb__learning_rate": [0.03, 0.1],
        "clf__passthrough": [True, False],
    }

    return {"RF": (rf, rf_space), "XGB": (xgb, xgb_space), "STK": (stk, stk_space)}

# ---------- 评估与落盘：返回字典，方便外面打印 ----------
def evaluate_and_save(cfg, name: str, y_true, y_prob, y_pred):
    os.makedirs(cfg.REPORT_DIR, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    try:
        if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        elif y_prob is not None:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(cfg.REPORT_DIR, f"cm_{name}.csv"), index=False)

    row = pd.DataFrame([{
        "model": name, "accuracy": acc, "auc_ovr": auc,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f1_w,
        "precision_macro": p_m,   "recall_macro": r_m,   "f1_macro": f1_m,
    }])
    out_path = os.path.join(cfg.REPORT_DIR, "metrics_classification.csv")
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(out_path, index=False)

    return {"model": name, "accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w, "auc_ovr": auc}

# ---------- 训练主入口：以 accuracy 选最佳 ----------
def train_all(cfg, df: pd.DataFrame):
    if cfg.LABEL_COL not in df.columns:
        raise KeyError(f"Label column {cfg.LABEL_COL!r} not found")
    y = df[cfg.LABEL_COL]
    X = df.drop(columns=[cfg.LABEL_COL])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = build_preprocessor(num_cols, cat_cols)

    skf = StratifiedKFold(n_splits=cfg.CLASS_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    spaces = model_spaces(cfg.RANDOM_SEED)
    best_name, best_score, best_model = None, -np.inf, None

    n_iter = getattr(cfg, "RAND_SEARCH_ITERS", 20)
    for name, (clf, space) in spaces.items():
        print(f"\n=== Training {name} (No resampling; class_weight if available) ===")
        pipe = Pipeline([("preprocess", pre), ("clf", clf)])
        search = RandomizedSearchCV(
            pipe, space, n_iter=n_iter, scoring=cfg.PRIMARY_SCORING,
            cv=skf, n_jobs=cfg.N_JOBS, verbose=1, refit=True, random_state=cfg.RANDOM_SEED
        )
        search.fit(X, y)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=cfg.RANDOM_SEED)
        best = search.best_estimator_
        best.fit(X_tr, y_tr)
        try:
            y_prob = best.predict_proba(X_te)
        except Exception:
            y_prob = None
        y_pred = best.predict(X_te)

        metrics = evaluate_and_save(cfg, name, y_te, y_prob, y_pred)
        print(f"[{name}] accuracy={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f} | f1_weighted={metrics['f1_weighted']:.4f}")

        # 以 accuracy 选最好
        acc = metrics["accuracy"]
        if acc > best_score:
            best_score, best_name, best_model = acc, name, best

    import joblib
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    best_path = os.path.join(cfg.MODEL_DIR, f"best_{best_name}.joblib")
    joblib.dump(best_model, best_path)
    print(f"Best model: {best_name} (accuracy={best_score:.4f}) -> {best_path}")
    return best_name
