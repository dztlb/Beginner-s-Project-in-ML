#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖包检查脚本
"""

def check_dependencies():
    """检查所有必要的依赖包"""
    print("正在检查依赖包...")
    
    # 基础包
    try:
        import pandas
        print("✓ pandas:", pandas.__version__)
    except ImportError as e:
        print("✗ pandas:", e)
        return False
    
    try:
        import numpy
        print("✓ numpy:", numpy.__version__)
    except ImportError as e:
        print("✗ numpy:", e)
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib:", matplotlib.__version__)
    except ImportError as e:
        print("✗ matplotlib:", e)
        return False
    
    try:
        import seaborn
        print("✓ seaborn:", seaborn.__version__)
    except ImportError as e:
        print("✗ seaborn:", e)
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("✗ scikit-learn:", e)
        return False
    
    try:
        import tensorflow
        print("✓ tensorflow:", tensorflow.__version__)
    except ImportError as e:
        print("✗ tensorflow:", e)
        return False
    
    try:
        import keras
        print("✓ keras:", keras.__version__)
    except ImportError as e:
        print("✗ keras:", e)
        return False
    
    try:
        import scipy
        print("✓ scipy:", scipy.__version__)
    except ImportError as e:
        print("✗ scipy:", e)
        return False
    
    try:
        import statsmodels
        print("✓ statsmodels:", statsmodels.__version__)
    except ImportError as e:
        print("✗ statsmodels:", e)
        return False
    
    # 可选包
    try:
        import pywt
        print("✓ PyWavelets:", pywt.__version__)
    except ImportError as e:
        print("⚠ PyWavelets: 未安装 (可选)")
    
    try:
        from PyEMD import CEEMDAN
        print("✓ PyEMD: 已安装")
    except ImportError as e:
        print("⚠ PyEMD: 未安装 (可选)")
    
    print("\n所有必要依赖包检查完成！")
    return True

if __name__ == "__main__":
    check_dependencies()
