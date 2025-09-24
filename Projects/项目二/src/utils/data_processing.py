#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理模块
包含数据加载、清洗、预处理等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
import warnings

def load_data(file_path, sample_size=None):
    """加载数据"""
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    print(f"数据加载完成，共 {len(df)} 行，{len(df.columns)} 列")
    
    # 如果指定了采样大小，则进行数据采样
    if sample_size and sample_size < len(df):
        print(f"为了快速验证，采样 {sample_size} 行数据进行测试...")
        # 均匀采样
        step = len(df) // sample_size
        df = df.iloc[::step].reset_index(drop=True)
        print(f"采样后数据量: {len(df)} 行")
    
    return df

def clean_data(df):
    """数据清洗"""
    print("正在进行数据清洗...")
    
    # 检查数据类型
    print("数据类型检查:")
    print(df.dtypes)
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 检查重复值
    print(f"\n重复行数量: {df.duplicated().sum()}")
    
    # 基本统计信息
    print("\n基本统计信息:")
    print(df.describe())
    
    print("数据清洗完成")
    return df

def sliding_window_imputation(df, column, window_size=5):
    """滑动窗口均值插补法"""
    df_imputed = df.copy()
    
    for i in range(len(df_imputed)):
        if pd.isna(df_imputed.loc[i, column]):
            # 获取前后窗口的数据
            start_idx = max(0, i - window_size)
            end_idx = min(len(df_imputed), i + window_size + 1)
            
            # 计算窗口内非缺失值的均值
            window_data = df_imputed.loc[start_idx:end_idx, column]
            window_data = window_data.dropna()
            
            if len(window_data) > 0:
                df_imputed.loc[i, column] = window_data.mean()
    
    return df_imputed

def detect_and_correct_outliers(df, target_col, feature_cols):
    """使用孤立森林检测异常值，SVM修正"""
    print("正在进行异常值检测与修正...")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])
    
    # 孤立森林异常检测
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(features_scaled)
    
    # 获取异常值索引
    outlier_indices = np.where(outliers == -1)[0]
    normal_indices = np.where(outliers == 1)[0]
    
    print(f"检测到 {len(outlier_indices)} 个异常值")
    
    if len(outlier_indices) > 0:
        # 使用SVM修正异常值
        svr = SVR(kernel='rbf')
        svr.fit(features_scaled[normal_indices], df.loc[normal_indices, target_col])
        
        # 预测异常值
        corrected_values = svr.predict(features_scaled[outlier_indices])
        
        # 更新异常值
        df_corrected = df.copy()
        df_corrected.loc[outlier_indices, target_col] = corrected_values
        
        print("异常值修正完成")
        return df_corrected
    
    print("未检测到异常值")
    return df
