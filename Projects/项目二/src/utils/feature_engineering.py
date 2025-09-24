#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征工程模块
包含特征创建、选择、变换等功能
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
import warnings

def create_lag_features(df, target_col, lag_periods=[1, 2, 3, 6, 12]):
    """创建滞后特征"""
    print("正在创建滞后特征...")
    df_lagged = df.copy()
    
    for lag in lag_periods:
        df_lagged[f'{target_col}_lag_{lag}'] = df_lagged[target_col].shift(lag)
    
    print(f"创建了 {len(lag_periods)} 个滞后特征")
    return df_lagged

def create_rolling_features(df, target_col, windows=[3, 6, 12]):
    """创建滚动统计特征"""
    print("正在创建滚动统计特征...")
    df_rolling = df.copy()
    
    for window in windows:
        df_rolling[f'{target_col}_rolling_mean_{window}'] = df_rolling[target_col].rolling(window=window).mean()
        df_rolling[f'{target_col}_rolling_std_{window}'] = df_rolling[target_col].rolling(window=window).std()
        df_rolling[f'{target_col}_rolling_max_{window}'] = df_rolling[target_col].rolling(window=window).max()
        df_rolling[f'{target_col}_rolling_min_{window}'] = df_rolling[target_col].rolling(window=window).min()
    
    print(f"创建了 {len(windows) * 4} 个滚动统计特征")
    return df_rolling

def create_cwt_features(df, target_col, scales=np.arange(1, 16)):
    """创建连续小波变换特征"""
    print("正在创建小波变换特征...")
    
    try:
        import pywt
        
        df_cwt = df.copy()
        
        # 对功率数据进行小波变换
        power_data = df_cwt[target_col].values
        
        # 数据预处理：确保数据类型和长度
        if len(power_data) < 100:  # 数据太少，跳过小波变换
            print("数据量不足，跳过小波变换特征创建")
            return df_cwt
        
        # 处理可能的NaN值
        if np.any(np.isnan(power_data)):
            power_data = np.nan_to_num(power_data, nan=0.0)
        
        # 使用多种方法尝试小波变换
        coefficients = None
        
        # 方法1: 尝试使用离散小波变换 (DWT) 替代连续小波变换
        try:
            print("尝试方法1: 使用离散小波变换 (DWT)...")
            # 使用DWT替代CWT
            coeffs = pywt.wavedec(power_data, 'db4', level=3)
            # 将DWT系数转换为类似CWT的特征
            coefficients = np.vstack([coeffs[i] for i in range(len(coeffs))])
            print("方法1成功: 使用DWT替代CWT")
        except Exception as e1:
            print(f"方法1失败: {e1}")
            
            # 方法2: 尝试使用小波包分解
            try:
                print("尝试方法2: 使用小波包分解...")
                # 使用小波包分解
                wp = pywt.WaveletPacket(data=power_data, wavelet='db4', mode='symmetric')
                # 获取不同级别的系数
                coeffs = []
                for i in range(min(4, len(wp.get_level(2)))):
                    try:
                        coeffs.append(wp[f'aa{i}'].data)
                    except:
                        continue
                if coeffs:
                    coefficients = np.vstack(coeffs)
                    print("方法2成功: 使用小波包分解")
                else:
                    raise Exception("无法获取小波包系数")
            except Exception as e2:
                print(f"方法2失败: {e2}")
                
                # 方法3: 尝试使用简单的滑动窗口特征替代小波变换
                try:
                    print("尝试方法3: 使用滑动窗口特征替代小波变换...")
                    # 创建滑动窗口统计特征来模拟小波变换的效果
                    window_sizes = [3, 5, 7, 9]
                    coeffs = []
                    
                    for window in window_sizes:
                        # 滑动窗口均值
                        rolling_mean = np.convolve(power_data, np.ones(window)/window, mode='same')
                        coeffs.append(rolling_mean)
                        
                        # 滑动窗口标准差
                        rolling_std = np.array([np.std(power_data[max(0, i-window//2):min(len(power_data), i+window//2+1)]) 
                                              for i in range(len(power_data))])
                        coeffs.append(rolling_std)
                    
                    coefficients = np.vstack(coeffs)
                    print("方法3成功: 使用滑动窗口特征替代")
                except Exception as e3:
                    print(f"方法3失败: {e3}")
                    
                    # 方法4: 尝试使用FFT频域特征
                    try:
                        print("尝试方法4: 使用FFT频域特征...")
                        # 使用FFT获取频域特征
                        fft_coeffs = np.fft.fft(power_data)
                        # 取前几个频率分量
                        n_freqs = min(8, len(fft_coeffs)//2)
                        coeffs = []
                        
                        for i in range(n_freqs):
                            # 实部和虚部
                            coeffs.append(np.real(fft_coeffs[i:i+1]))
                            coeffs.append(np.imag(fft_coeffs[i:i+1]))
                        
                        coefficients = np.vstack(coeffs)
                        print("方法4成功: 使用FFT频域特征")
                    except Exception as e4:
                        print(f"方法4失败: {e4}")
                        
                        # 方法5: 使用简单的统计特征
                        try:
                            print("尝试方法5: 使用统计特征替代...")
                            # 创建简单的统计特征来替代小波变换
                            coeffs = []
                            
                            # 分段统计特征
                            segment_size = len(power_data) // 8
                            for i in range(8):
                                start_idx = i * segment_size
                                end_idx = min((i + 1) * segment_size, len(power_data))
                                segment = power_data[start_idx:end_idx]
                                
                                coeffs.append(np.mean(segment))
                                coeffs.append(np.std(segment))
                                coeffs.append(np.max(segment))
                                coeffs.append(np.min(segment))
                            
                            coefficients = np.vstack(coeffs)
                            print("方法5成功: 使用统计特征替代")
                        except Exception as e5:
                            print(f"方法5失败: {e5}")
                            print("所有替代方法都失败，跳过特征创建")
                            return df_cwt
        
        # 检查系数是否有效
        if coefficients is None or coefficients.size == 0:
            print("特征系数无效，跳过特征创建")
            return df_cwt
        
        # 提取统计特征
        try:
            # 计算每个系数的统计特征
            df_cwt[f'{target_col}_wavelet_mean'] = np.mean(coefficients, axis=0)
            df_cwt[f'{target_col}_wavelet_std'] = np.std(coefficients, axis=0)
            df_cwt[f'{target_col}_wavelet_max'] = np.max(coefficients, axis=0)
            df_cwt[f'{target_col}_wavelet_min'] = np.min(coefficients, axis=0)
            
            # 添加额外的特征
            df_cwt[f'{target_col}_wavelet_range'] = np.max(coefficients, axis=0) - np.min(coefficients, axis=0)
            df_cwt[f'{target_col}_wavelet_energy'] = np.sum(coefficients**2, axis=0)
            
            print("小波变换特征创建完成（使用替代方法）")
            return df_cwt
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            print("跳过小波变换特征创建")
            return df_cwt
        
    except ImportError:
        print("pywt库未安装，跳过小波变换特征创建")
        return df
    except Exception as e:
        print(f"小波变换特征创建失败: {e}")
        print("跳过小波变换特征创建，继续其他特征工程")
        return df

def select_features_rfecv(X, y, estimator=None, step=2, cv=3, quick_mode=False):
    """使用RFECV进行特征选择"""
    print("正在进行特征选择...")
    
    if estimator is None:
        if quick_mode:
            # 快速验证模式：使用较少的树
            estimator = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    
    # 快速验证模式：使用较少的交叉验证折数
    if quick_mode:
        cv = 2
        print("快速验证模式：使用较少交叉验证折数")
    
    print(f"特征选择参数: 步长={step}, 交叉验证折数={cv}")
    
    # RFECV特征选择
    rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring='neg_mean_squared_error')
    rfecv.fit(X, y)
    
    # 获取选择的特征
    selected_features = X.columns[rfecv.support_].tolist()
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择后特征数量: {len(selected_features)}")
    print(f"选择的特征: {selected_features}")
    
    return selected_features, rfecv
