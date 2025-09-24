"""
数据预处理模块测试
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """数据预处理器测试类"""
    
    def test_load_data(self):
        """测试数据加载功能"""
        preprocessor = DataPreprocessor('data/data_aftercalculate.xlsx')
        df = preprocessor.load_data()
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_clean_data(self):
        """测试数据清洗功能"""
        preprocessor = DataPreprocessor('data/data_aftercalculate.xlsx')
        df = preprocessor.load_data()
        cleaned_df = preprocessor.clean_data()
        
        assert cleaned_df is not None
        assert isinstance(cleaned_df, pd.DataFrame)
    
    def test_encode_categorical_features(self):
        """测试分类特征编码功能"""
        preprocessor = DataPreprocessor('data/data_aftercalculate.xlsx')
        df = preprocessor.load_data()
        df = preprocessor.clean_data()
        encoded_df = preprocessor.encode_categorical_features()
        
        assert encoded_df is not None
        assert isinstance(encoded_df, pd.DataFrame)
    
    def test_prepare_features(self):
        """测试特征准备功能"""
        preprocessor = DataPreprocessor('data/data_aftercalculate.xlsx')
        df = preprocessor.load_data()
        df = preprocessor.clean_data()
        df = preprocessor.encode_categorical_features()
        X, y = preprocessor.prepare_features()
        
        assert X is not None
        assert y is not None
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
