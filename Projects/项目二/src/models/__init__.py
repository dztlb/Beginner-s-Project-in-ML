"""
模型训练包
包含深度学习模型和传统机器学习模型的训练功能
"""

from .deep_learning import train_deep_learning_model
from .random_forest import train_random_forest

__all__ = [
    'train_deep_learning_model',
    'train_random_forest'
]
