"""
项目配置文件
"""

import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据配置
DATA_CONFIG = {
    'sequence_length': 5,
    'prediction_length': 20,
    'batch_size': 8,  # 减小批次大小以节省内存
    'num_workers': 2,  # 减少工作进程以节省内存
    'frame_interval_minutes': 6,  # 帧间时间分辨率(分钟)，例如6分钟/帧 → 20帧≈2小时
    'use_simple_dataset': True,
    'metadata_path': os.path.join(PROJECT_ROOT, 'data', 'raw', '项目4-hdf_metadata.csv'),
    'data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'taasrad19_dir': os.path.join(PROJECT_ROOT, 'data', 'TAASRAD19'),
    'synthetic_data_dir': os.path.join(PROJECT_ROOT, 'synthetic_data'),
    'normalized_data_path': os.path.join(PROJECT_ROOT, 'data', 'normalized_data.hdf')
}

# 训练配置
TRAINING_CONFIG = {
    'num_epochs': 50,
    'learning_rate': 0.001,
    'device': 'cuda',  # 'cuda' 或 'cpu'
    'save_dir': os.path.join(PROJECT_ROOT, 'checkpoints'),
    'model_type': 'enhanced_simam_res_unet',  # 'simam_unet', 'simam_res_unet', 'simam_unet_sequence', 'simam_res_unet_sequence', 'enhanced_simam_res_unet', 'enhanced_simam_res_unet_sequence'
    'save_interval': 10,
    'early_stopping_patience': 15,
    'weight_decay': 1e-5
}

# 模型配置
MODEL_CONFIG = {
    'features': [32, 64, 128, 256],  # 减小特征数以避免内存问题
    'bilinear': True,  # 使用双线性插值避免通道数问题
    'dropout': 0.2,
    'activation': 'relu',
    'use_depthwise': True  # 是否启用深度可分离卷积
}

# 路径配置
PATHS = {
    'project_root': PROJECT_ROOT,
    'data_dir': os.path.join(PROJECT_ROOT, 'data'),
    'checkpoints_dir': os.path.join(PROJECT_ROOT, 'checkpoints'),
    'logs_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    'synthetic_data_dir': os.path.join(PROJECT_ROOT, 'synthetic_data'),
    'taasrad19_dir': os.path.join(PROJECT_ROOT, 'data', 'TAASRAD19')
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (12, 5),
    'dpi': 300,
    'save_format': 'png',
    'font_size': 12,
    'num_samples': 3  # 保存可视化的样本数量
}

# 测试配置
TEST_CONFIG = {
    'sequence_length': 5,
    'prediction_length': 20,
    'batch_size': 8,
    'device': 'cuda',
    'model_type': 'simam_unet',
    'data_path': os.path.join(PROJECT_ROOT, 'data', 'normalized_data.hdf'),
    'test_metadata_path': os.path.join(PROJECT_ROOT, 'data', 'test_data.csv'),
    'model_path': os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth'),
    'results_dir': os.path.join(PROJECT_ROOT, 'results')
}

# 数据预处理配置
PREPROCESSING_CONFIG = {
    'input_size': (256, 256),
    'normalization_method': 'minmax',  # 'minmax', 'zscore', 'robust'
    'interpolation_method': 'bilinear',
    'data_format': 'hdf5'
}

