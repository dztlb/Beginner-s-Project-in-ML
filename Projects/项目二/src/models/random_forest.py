#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机森林模型训练模块
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model_utils import AdaptiveSardineSwarmOptimization
from visualization import plot_optimization_convergence

def train_random_forest(X_train, y_train, X_test, y_test, quick_mode=False):
    """训练随机森林模型（自适应沙丁鱼群优化）"""
    print("开始随机森林模型训练（自适应沙丁鱼群优化）...")
    
    # 根据模式调整搜索空间和优化参数
    if quick_mode:
        # 快速验证模式：使用较小的搜索空间和较少的迭代
        search_space = {
            'n_estimators': (20, 50, 'int'),
            'max_depth': (3, 10, 'int'),
            'min_samples_split': (2, 8, 'int'),
            'min_samples_leaf': (1, 4, 'int')
        }
        
        sardine_optimizer = AdaptiveSardineSwarmOptimization(
            n_sardines=8,  # 较少的沙丁鱼
            max_iterations=10,  # 较少的迭代次数
            search_space=search_space
        )
        print("快速验证模式：使用较小搜索空间和较少迭代")
    else:
        # 完整训练模式：使用完整的搜索空间
        search_space = {
            'n_estimators': (50, 200, 'int'),
            'max_depth': (5, 30, 'int'),
            'min_samples_split': (2, 15, 'int'),
            'min_samples_leaf': (1, 8, 'int')
        }
        
        sardine_optimizer = AdaptiveSardineSwarmOptimization(
            n_sardines=15,
            max_iterations=20,
            search_space=search_space
        )
    
    try:
        print("开始参数优化...")
        best_params, best_fitness, fitness_history = sardine_optimizer.optimize(
            X_train, y_train, X_test, y_test
        )
        
        print(f"优化完成！最佳参数: {best_params}")
        print(f"最佳适应度: {best_fitness:.4f}")
        
        # 使用最佳参数训练最终模型
        print("使用最佳参数训练最终模型...")
        final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=1)
        final_model.fit(X_train, y_train)
        print("最终模型训练完成！")
        
        # 预测和评估
        print("进行预测和评估...")
        y_pred_train = final_model.predict(X_train)
        y_pred_test = final_model.predict(X_test)
        
        # 计算评估指标
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("\n随机森林模型评估结果:")
        print(f"训练集 - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
        print(f"测试集 - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
        
        # 绘制优化收敛曲线
        plot_optimization_convergence(fitness_history)
        
        return final_model, best_params, best_fitness, fitness_history, test_mse, test_rmse, test_mae, test_r2
        
    except Exception as e:
        print(f"随机森林训练过程中出错: {e}")
        print("使用默认参数训练随机森林模型...")
        
        # 使用默认参数作为备选方案
        default_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        default_model.fit(X_train, y_train)
        
        y_pred_train = default_model.predict(X_train)
        y_pred_test = default_model.predict(X_test)
        
        # 计算评估指标
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"默认模型测试集结果 - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")
        
        return default_model, {}, 0, [], test_mse, test_rmse, test_mae, test_r2
