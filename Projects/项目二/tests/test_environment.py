#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境配置验证测试文件
用于验证Anaconda和PyCharm环境是否正确配置
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_basic_imports():
    """测试基本包的导入"""
    print("=" * 60)
    print("环境配置验证测试")
    print("=" * 60)
    
    # 测试Python版本
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print()
    
    # 测试基本包导入
    try:
        import pandas as pd
        print(f"✓ Pandas版本: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ Numpy版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ Numpy导入失败: {e}")
        return False
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib版本: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib导入失败: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn版本: {sns.__version__}")
    except ImportError as e:
        print(f"✗ Seaborn导入失败: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn版本: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn导入失败: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow版本: {tf.__version__}")
        print(f"  GPU可用: {tf.config.list_physical_devices('GPU')}")
    except ImportError as e:
        print(f"✗ TensorFlow导入失败: {e}")
        return False
    
    try:
        import joblib
        print(f"✓ Joblib版本: {joblib.__version__}")
    except ImportError as e:
        print(f"✗ Joblib导入失败: {e}")
        return False
    
    return True

def test_data_operations():
    """测试数据处理功能"""
    print("\n" + "=" * 60)
    print("数据处理功能测试")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        }
        df = pd.DataFrame(data)
        print(f"✓ 成功创建DataFrame，形状: {df.shape}")
        
        # 测试基本统计
        print(f"✓ 数据统计: 均值={df['feature1'].mean():.4f}, 标准差={df['feature1'].std():.4f}")
        
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        return False
    
    return True

def test_ml_operations():
    """测试机器学习功能"""
    print("\n" + "=" * 60)
    print("机器学习功能测试")
    print("=" * 60)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        # 创建测试数据
        import numpy as np
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✓ 成功划分数据集: 训练集{X_train.shape}, 测试集{X_test.shape}")
        
        # 训练简单模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        print(f"✓ 成功训练随机森林模型，MSE: {mse:.4f}")
        
    except Exception as e:
        print(f"✗ 机器学习测试失败: {e}")
        return False
    
    return True

def test_deep_learning():
    """测试深度学习功能"""
    print("\n" + "=" * 60)
    print("深度学习功能测试")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        
        # 创建简单模型
        model = Sequential([
            Dense(10, activation='relu', input_shape=(5,)),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        print("✓ 成功创建TensorFlow模型")
        print(f"  模型参数数量: {model.count_params()}")
        
        # 测试模型预测
        import numpy as np
        test_input = np.random.randn(1, 5)
        prediction = model.predict(test_input)
        print(f"✓ 成功进行模型预测，输出形状: {prediction.shape}")
        
    except Exception as e:
        print(f"✗ 深度学习测试失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始环境配置验证测试...")
    print()
    
    # 运行各项测试
    tests = [
        ("基本包导入", test_basic_imports),
        ("数据处理", test_data_operations),
        ("机器学习", test_ml_operations),
        ("深度学习", test_deep_learning)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name}测试失败")
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"总测试数: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境配置成功！")
        print("现在可以运行主程序 wind_power_prediction.py")
    else:
        print("\n❌ 部分测试失败，请检查环境配置")
        print("建议:")
        print("1. 确认conda环境已正确激活")
        print("2. 检查PyCharm中的Python解释器设置")
        print("3. 重新安装失败的包")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
