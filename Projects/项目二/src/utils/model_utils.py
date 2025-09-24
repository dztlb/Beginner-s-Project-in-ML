#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型工具模块
包含深度学习模型构建、训练等功能
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def create_tcn_attention_bigru_model(input_shape, num_features):
    """构建TCN-多头注意力-BiGRU模型"""
    
    # 输入层
    inputs = keras.Input(shape=input_shape)
    
    # TCN层 (替代CNN)
    def tcn_block(x, filters, kernel_size, dilation_rate):
        """TCN块"""
        # 因果卷积 - 使用same padding保持时间维度
        conv = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='same',
            activation='relu'
        )(x)
        
        # 批归一化
        conv = keras.layers.BatchNormalization()(conv)
        
        # 残差连接 - 确保维度匹配
        if x.shape[-1] != filters:
            x = keras.layers.Conv1D(filters, 1, padding='same')(x)
        
        # 确保时间维度一致
        if x.shape[1] != conv.shape[1]:
            # 如果时间维度不匹配，使用全局平均池化然后重复
            x = keras.layers.GlobalAveragePooling1D()(x)
            x = keras.layers.RepeatVector(conv.shape[1])(x)
        
        return keras.layers.Add()([x, conv])
    
    # TCN特征提取
    x = inputs
    for i in range(3):  # 3个TCN块
        x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2**i)
    
    # 简化的注意力机制 - 使用Keras层而不是TensorFlow操作
    # 使用全局平均池化和全局最大池化来捕获时间信息
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    
    # 合并池化结果
    pooled = keras.layers.Concatenate()([avg_pool, max_pool])
    
    # 注意力权重
    attention_weights = keras.layers.Dense(128, activation='relu')(pooled)
    attention_weights = keras.layers.Dense(64, activation='sigmoid')(attention_weights)
    
    # 应用注意力到原始特征
    # 使用Lambda层来应用注意力权重
    def apply_attention(args):
        features, weights = args
        # 扩展权重维度以匹配特征维度
        weights_expanded = keras.backend.expand_dims(weights, axis=1)
        # 重复权重以匹配时间步长
        weights_repeated = keras.backend.repeat_elements(weights_expanded, features.shape[1], axis=1)
        return features * weights_repeated
    
    attended_features = keras.layers.Lambda(apply_attention)([x, attention_weights])
    
    # BiGRU层 - 输入是3D (batch_size, timesteps, features)
    bigru = keras.layers.Bidirectional(
        keras.layers.GRU(128, return_sequences=True)
    )(attended_features)
    
    bigru = keras.layers.Bidirectional(
        keras.layers.GRU(64, return_sequences=False)
    )(bigru)
    
    # 全连接层
    dense = keras.layers.Dense(128, activation='relu')(bigru)
    dense = keras.layers.Dropout(0.3)(dense)
    dense = keras.layers.Dense(64, activation='relu')(dense)
    dense = keras.layers.Dropout(0.2)(dense)
    
    # 输出层
    outputs = keras.layers.Dense(1, activation='linear')(dense)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_sequences(X, y, time_steps=10):
    """创建时间序列数据"""
    Xs, ys = [], []
    
    # 确保数据长度足够
    if len(X) <= time_steps:
        raise ValueError(f"数据长度({len(X)})必须大于时间步长({time_steps})")
    
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    
    # 转换为numpy数组
    Xs = np.array(Xs)
    ys = np.array(ys)
    
    # 检查输出形状
    print(f"序列创建完成 - X形状: {Xs.shape}, y形状: {ys.shape}")
    
    return Xs, ys

class AdaptiveSardineSwarmOptimization:
    """自适应沙丁鱼群优化算法"""
    def __init__(self, n_sardines=15, max_iterations=20, search_space=None):
        self.n_sardines = n_sardines
        self.max_iterations = max_iterations
        self.search_space = search_space or {}
        self.best_position = None
        self.best_fitness = float('inf')
        self.fitness_history = []
    
    def initialize_population(self):
        """初始化沙丁鱼群"""
        population = []
        for _ in range(self.n_sardines):
            sardine = {}
            for param, (min_val, max_val, param_type) in self.search_space.items():
                if param_type == 'int':
                    sardine[param] = np.random.randint(min_val, max_val + 1)
                else:
                    sardine[param] = np.random.uniform(min_val, max_val)
            population.append(sardine)
        return population
    
    def evaluate_fitness(self, sardine, X_train, y_train, X_val, y_val):
        """评估适应度"""
        try:
            # 使用较小的n_estimators进行快速评估
            test_params = sardine.copy()
            if 'n_estimators' in test_params:
                test_params['n_estimators'] = min(test_params['n_estimators'], 50)
            
            model = RandomForestRegressor(**test_params, random_state=42, n_jobs=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        except Exception as e:
            print(f"评估适应度时出错: {e}")
            return float('inf')
    
    def update_position(self, sardine, best_sardine, iteration):
        """更新沙丁鱼位置"""
        alpha = 0.5 * (1 - iteration / self.max_iterations)  # 自适应学习因子
        beta = 0.3 * (iteration / self.max_iterations)  # 自适应探索因子
        
        for param in sardine.keys():
            if param in best_sardine:
                # 向最优解移动
                sardine[param] += alpha * (best_sardine[param] - sardine[param])
                
                # 添加随机探索
                sardine[param] += beta * np.random.normal(0, 1)
                
                # 确保参数在有效范围内
                min_val, max_val, param_type = self.search_space[param]
                sardine[param] = np.clip(sardine[param], min_val, max_val)
                
                # 整数参数取整
                if param_type == 'int':
                    sardine[param] = int(sardine[param])
        
        return sardine
    
    def optimize(self, X_train, y_train, X_val, y_val):
        """主优化循环"""
        print(f"开始优化，种群大小: {self.n_sardines}, 最大迭代次数: {self.max_iterations}")
        population = self.initialize_population()
        
        for iteration in range(self.max_iterations):
            print(f"迭代 {iteration + 1}/{self.max_iterations}...")
            
            # 评估适应度
            for i, sardine in enumerate(population):
                if i % 5 == 0:  # 每5个沙丁鱼显示一次进度
                    print(f"  评估沙丁鱼 {i + 1}/{self.n_sardines}")
                
                fitness = self.evaluate_fitness(sardine, X_train, y_train, X_val, y_val)
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = sardine.copy()
                    print(f"  发现更好的解: {fitness:.4f}")
            
            # 更新位置
            for sardine in population:
                sardine = self.update_position(sardine, self.best_position, iteration)
            
            # 记录历史
            self.fitness_history.append(self.best_fitness)
            
            print(f"迭代 {iteration + 1}: 最佳适应度 = {self.best_fitness:.4f}")
        
        return self.best_position, self.best_fitness, self.fitness_history
