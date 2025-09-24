#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块
包含图表绘制、结果展示等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from statsmodels.tsa.stattools import kpss
import warnings

# 设置matplotlib使用非交互式后端，避免显示问题
plt.switch_backend('Agg')

# 尝试设置支持中文的字体
try:
    # 获取系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 优先使用支持中文的字体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    font_found = False
    
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            font_found = True
            print(f"找到并使用中文字体: {font_name}")
            break
    
    if not font_found:
        # 如果没有中文字体，使用英文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("未找到中文字体，使用英文显示")
    
    plt.rcParams['axes.unicode_minus'] = False
    
except Exception as e:
    print(f"字体设置失败: {e}")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def analyze_correlations(df):
    """相关性分析"""
    print("正在进行相关性分析...")
    
    # 选择数值列进行相关性分析
    numeric_cols = ['WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WINDSPEED', 'POWER']
    
    # 计算皮尔逊相关系数
    correlation_matrix = df[numeric_cols].corr()
    
    # 绘制相关性热力图
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8}, annot_kws={'size': 12})
    plt.title('Feature Correlation Heatmap', fontsize=20, pad=30)
    plt.tight_layout()
    plt.savefig('results/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算与功率的相关系数
    power_correlations = correlation_matrix['POWER'].sort_values(ascending=False)
    print("\n与功率的相关系数:")
    for feature, corr in power_correlations.items():
        if feature != 'POWER':
            print(f"{feature}: {corr:.4f}")
    
    return correlation_matrix

def kpss_stationarity_test(df):
    """KPSS平稳性检验"""
    print("正在进行KPSS平稳性检验...")
    
    power_data = df['POWER'].dropna()
    
    try:
        # 抑制KPSS警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # KPSS检验
            kpss_stat, p_value, lags, critical_values = kpss(power_data, regression='c')
        
        print(f"KPSS统计量: {kpss_stat:.4f}")
        print(f"P值: {p_value:.4f}")
        print(f"临界值: {critical_values}")
        
        # 更详细的解释
        if kpss_stat > critical_values['5%']:
            print("结论: 功率数据非平稳 (KPSS统计量 > 5%临界值)")
            if kpss_stat > critical_values['1%']:
                print("      数据高度非平稳 (KPSS统计量 > 1%临界值)")
        else:
            print("结论: 功率数据平稳 (KPSS统计量 < 5%临界值)")
        
        # 如果p值很小，说明数据非常非平稳
        if p_value < 0.001:
            print("注意: P值极小，表明数据具有强烈的非平稳性")
        
        return kpss_stat, p_value
        
    except Exception as e:
        print(f"KPSS检验失败: {e}")
        return None, None

def plot_results_comparison(rf_metrics, dl_metrics):
    """绘制模型对比结果"""
    print("正在绘制模型对比结果...")
    
    # 创建对比表格
    comparison_data = {
        'Model': ['Random Forest', 'TCN-Attention-BiGRU'],
        'MSE': [rf_metrics[0], dl_metrics[0]],
        'RMSE': [rf_metrics[1], dl_metrics[1]],
        'MAE': [rf_metrics[2], dl_metrics[2]],
        'R2': [rf_metrics[3], dl_metrics[3]]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n模型性能对比:")
    print(comparison_df)
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    metric_labels = ['MSE', 'RMSE', 'MAE', 'R2']
    colors = ['skyblue', 'lightcoral']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row, col = i // 2, i % 2
        bars = axes[row, col].bar(comparison_df['Model'], comparison_df[metric], color=colors, width=0.6)
        axes[row, col].set_title(f'{label} Comparison', fontsize=16, pad=20)
        axes[row, col].set_ylabel(label, fontsize=14)
        axes[row, col].tick_params(axis='both', which='major', labelsize=12)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:.4f}', ha='center', va='bottom', fontsize=11)
        
        if metric == 'R2':
            axes[row, col].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, save_path='results/plots/training_history.png'):
    """绘制训练历史"""
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model MAE', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.legend(['Training MAE', 'Validation MAE'], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimization_convergence(fitness_history, save_path='results/plots/optimization_convergence.png'):
    """绘制优化收敛曲线"""
    plt.figure(figsize=(16, 8))
    plt.plot(fitness_history, linewidth=3, color='blue', marker='o', markersize=6)
    plt.title('Adaptive Sardine Swarm Optimization Convergence', fontsize=18)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Best Fitness (MSE)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
