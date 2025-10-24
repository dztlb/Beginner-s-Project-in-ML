# Quick dependency check
import importlib

pkgs = ["pandas","numpy","matplotlib","sklearn","statsmodels","scipy","pywt"]
for p in pkgs:
    try:
        importlib.import_module(p)
        print(f"OK: {p}")
    except Exception as e:
        print(f"FAIL: {p} - {e}")

try:
    import tensorflow as tf
    print("OK: tensorflow", tf.__version__)
except Exception:
    print("Optional: tensorflow not installed")
