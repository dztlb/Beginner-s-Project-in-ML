# -*- coding: utf-8 -*-
"""
配置文件 - 管理数据集路径和其他配置参数
"""

# 数据集路径配置
DATASET_PATHS = {
    'main_dataset': 'data/data_aftercalculate.xlsx',
    'literature_dataset': 'data/数据集来源文献.xlsx'
}

# 模型配置
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# 特征选择配置
FEATURE_SELECTION_CONFIG = {
    'n_features_to_select': 8,
    'cv_folds': 5
}

# 输出目录配置
OUTPUT_DIRS = {
    'models': 'models',
    'results': 'results',
    'plots': 'plots',
    'docs': 'docs',
    'scripts': 'scripts'
}

# 超参数调优配置
HYPERPARAMETER_CONFIG = {
    'RF': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'XGB': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    }
}
