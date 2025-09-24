#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度学习模型训练模块
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from model_utils import create_tcn_attention_bigru_model, create_sequences
from visualization import plot_training_history

def train_deep_learning_model(X_train, y_train, X_test, y_test, selected_features, quick_mode=False):
    """训练深度学习模型"""
    print("正在训练深度学习模型...")
    
    try:
        # 使用选择的特征
        X_selected = X_train[selected_features].fillna(0)
        X_test_selected = X_test[selected_features].fillna(0)
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # 创建时间序列数据
        time_steps = 10
        X_train_seq, y_train_seq = create_sequences(X_scaled, y_train.values, time_steps)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, time_steps)
        
        print(f"训练集形状: {X_train_seq.shape}")
        print(f"测试集形状: {X_test_seq.shape}")
        
        # 检查数据形状
        if len(X_train_seq.shape) != 3:
            raise ValueError(f"训练数据应该是3D形状，当前形状: {X_train_seq.shape}")
        if len(X_test_seq.shape) != 3:
            raise ValueError(f"测试数据应该是3D形状，当前形状: {X_test_seq.shape}")
        
        # 构建模型
        model = create_tcn_attention_bigru_model(
            input_shape=(time_steps, X_train_seq.shape[2]),
            num_features=X_train_seq.shape[2]
        )
        
        print("模型构建成功！")
        print(f"模型输入形状: {model.input_shape}")
        print(f"模型输出形状: {model.output_shape}")
        
        # 打印模型摘要
        print("\n模型结构摘要:")
        model.summary()
        
    except Exception as e:
        print(f"模型构建失败: {e}")
        raise
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # 早停机制
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 学习率调度
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # 根据模式调整训练参数
    if quick_mode:
        epochs = 20  # 快速验证模式使用较少的epochs
        batch_size = 16  # 较小的batch_size
        print("快速验证模式：使用较少训练轮次")
    else:
        epochs = 100  # 完整训练模式
        batch_size = 32
    
    print(f"训练参数: epochs={epochs}, batch_size={batch_size}, time_steps={time_steps}")
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    # 预测
    y_pred_dl = model.predict(X_test_seq)
    
    # 计算评估指标
    dl_mse = mean_squared_error(y_test_seq, y_pred_dl)
    dl_rmse = np.sqrt(dl_mse)
    dl_mae = mean_absolute_error(y_test_seq, y_pred_dl)
    dl_r2 = r2_score(y_test_seq, y_pred_dl)
    
    print("\n深度学习模型评估结果:")
    print(f"MSE: {dl_mse:.4f}")
    print(f"RMSE: {dl_rmse:.4f}")
    print(f"MAE: {dl_mae:.4f}")
    print(f"R2: {dl_r2:.4f}")
    
    # 绘制训练历史
    plot_training_history(history)
    
    return model, dl_mse, dl_rmse, dl_mae, dl_r2, history
