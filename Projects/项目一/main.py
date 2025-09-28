"""
Classification-only quick runner (NO cleaning): loads preprocessed Excel and trains models.
"""
import os
import pandas as pd

import config as cfg
from src.modeling import train_all

def main():
    xlsx = cfg.INPUT_XLSX
    print(f'[1/2] Loading preprocessed Excel: {xlsx}')
    df = pd.read_excel(xlsx)

    for d in [cfg.REPORT_DIR, cfg.FIG_DIR, cfg.MODEL_DIR, cfg.ARTIFACT_DIR]:
        os.makedirs(d, exist_ok=True)

    print('[2/2] Training & evaluating models...')
    best = train_all(cfg, df)
    print(f'Done. Best model: {best}')

if __name__ == '__main__':
    main()
