#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from src.main import run_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="Wind power prediction full pipeline")
    p.add_argument("--csv", required=True, help="Path to CSV file")
    p.add_argument("--datetime", default="DATETIME", help="Datetime column name")
    p.add_argument("--target", default="POWER", help="Target column name")
    p.add_argument("--save_prefix", default="run1", help="Prefix for outputs")
    p.add_argument("--time_steps", type=int, default=10, help="Sequence length for DL model")
    p.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        csv_path=args.csv,
        datetime_col=args.datetime,
        target_col=args.target,
        save_prefix=args.save_prefix,
        time_steps=args.time_steps,
        test_size=args.test_size,
    )
