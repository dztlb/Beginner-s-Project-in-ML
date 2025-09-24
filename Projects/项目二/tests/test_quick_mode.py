#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证模式测试版本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_data():
    """加载和处理数据"""
    print("正在加载数据...")
    df = pd.read_csv('项目2-0.csv')
    print(f"数据加载完成，共 {len(df)} 行")
    
    # 快速验证模式：只使用1000行数据
    df = df.iloc[::len(df)//1000].reset_index(drop=True)
    print(f"快速验证模式：使用 {len(df)} 行数据")
    
    # 处理缺失值
    df = df.fillna(df.mean(numeric_only=True))
    
    # 创建简单特征
    df['POWER_lag_1'] = df['POWER'].shift(1)
    df['WINDSPEED_squared'] = df['WINDSPEED'] ** 2
    
    # 去除缺失值
    df = df.dropna()
    
    return df

def train_simple_models(df):
    """训练简单模型"""
    print("准备训练数据...")
    
    # 特征和目标
    feature_cols = ['WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDSPEED', 'POWER_lag_1', 'WINDSPEED_squared']
    X = df[feature_cols]
    y = df['POWER']
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("训练随机森林模型...")
    # 随机森林模型
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred_rf = rf.predict(X_test)
    
    # 评估
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    print("\n随机森林模型结果:")
    print(f"MSE: {rf_mse:.4f}")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"MAE: {rf_mae:.4f}")
    print(f"R2: {rf_r2:.4f}")
    
    return rf, rf_mse, rf_rmse, rf_mae, rf_r2

def main():
    """主函数"""
    print("=" * 50)
    print("🚀 快速验证模式测试")
    print("=" * 50)
    
    try:
        # 加载数据
        df = load_and_process_data()
        
        # 训练模型
        rf_model, rf_mse, rf_rmse, rf_mae, rf_r2 = train_simple_models(df)
        
        print("\n" + "=" * 50)
        print("🎉 快速验证完成！")
        print("✅ 代码验证成功，随机森林模型正常运行")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

