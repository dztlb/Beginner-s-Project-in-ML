# -*- coding: utf-8 -*-
import os, json, random
import numpy as np

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.asarray(Xs), np.asarray(ys)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def seed_everything(seed=42):
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
