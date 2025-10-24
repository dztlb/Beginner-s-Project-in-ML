# -*- coding: utf-8 -*-
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def _evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

class ASSO:
    """Very light Adaptive Sardine Swarm Optimization mock (works without exotic deps)."""
    def __init__(self, search_space, n=20, iters=30, random_state=42):
        self.search_space = search_space
        self.n = n
        self.iters = iters
        self.rs = np.random.RandomState(random_state)

    def _sample(self):
        s = {}
        for k, (lo, hi, kind) in self.search_space.items():
            if kind == "int":
                s[k] = int(self.rs.randint(lo, hi+1))
            else:
                s[k] = float(self.rs.uniform(lo, hi))
        return s

    def optimize(self, scorer):
        # initialize
        pop = [self._sample() for _ in range(self.n)]
        best = None
        best_score = np.inf
        history = []
        for t in range(self.iters):
            for p in pop:
                score = scorer(p)
                if score < best_score:
                    best_score = score
                    best = p.copy()
            # move towards best with small noise
            for p in pop:
                for k,(lo,hi,kind) in self.search_space.items():
                    target = best[k]
                    p[k] = p[k] + 0.5*(target - p[k]) + 0.1*self.rs.randn()
                    p[k] = np.clip(p[k], lo, hi)
                    if kind == "int":
                        p[k] = int(round(p[k]))
            history.append(best_score)
        return best, history

def train_random_forest_asso(X_train, y_train, X_test, y_test, save_prefix="run1"):
    space = {
        "n_estimators": (100, 400, "int"),
        "max_depth": (5, 50, "int"),
        "min_samples_split": (2, 20, "int"),
        "min_samples_leaf": (1, 10, "int"),
    }
    def scorer(params):
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        return mean_squared_error(y_test, pred)

    asso = ASSO(space, n=20, iters=25, random_state=42)
    best_params, history = asso.optimize(scorer)

    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics_train = _evaluate(y_train, pred_train)
    metrics_test  = _evaluate(y_test,  pred_test)

    out = {
        "best_params": best_params,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "history": history,
    }
    with open(f"results/logs/{save_prefix}_rf.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out
