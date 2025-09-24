"""
数据预处理模块
用于加载、清洗和准备MgCo2O4电极材料数据
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self, data_path):
        """
        初始化数据预处理器
        
        Parameters:
        -----------
        data_path : str
            数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """加载Excel数据"""
        try:
            self.df = pd.read_excel(self.data_path)
            print(f"数据加载成功！数据形状: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def clean_data(self):
        """数据清洗"""
        if self.df is None:
            print("请先加载数据！")
            return None
            
        # 删除无用列
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)
            print("删除无用列 'Unnamed: 0'")
        
        # 处理缺失值
        missing_counts = self.df.isnull().sum()
        print(f"缺失值统计:\n{missing_counts[missing_counts > 0]}")
        
        # 用中位数填充数值型缺失值
        numeric_columns = ['c-time', 'c-temperature', 'load mass-mg/cm2']
        for col in numeric_columns:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"列 '{col}' 缺失值已用中位数 {median_val:.2f} 填充")
        
        print("数据清洗完成！")
        return self.df
    
    def handle_rare_classes(self, target_column='Secondary Morphology', min_count=2):
        """处理稀有类别，将样本数少于min_count的类别合并为'Other'"""
        if self.df is None:
            print("请先加载数据！")
            return None
            
        if target_column not in self.df.columns:
            print(f"目标列 '{target_column}' 不存在！")
            return None
            
        # 统计各类别样本数
        from collections import Counter
        class_counts = Counter(self.df[target_column])
        
        # 找出稀有类别
        rare_classes = [cls for cls, count in class_counts.items() if count < min_count]
        
        if rare_classes:
            print(f"发现 {len(rare_classes)} 个稀有类别（样本数 < {min_count}）:")
            for cls in rare_classes:
                print(f"  - {cls}: {class_counts[cls]} 样本")
            
            # 将稀有类别合并为'Other'
            self.df[target_column] = self.df[target_column].apply(
                lambda x: 'Other' if x in rare_classes else x
            )
            
            print(f"已将稀有类别合并为 'Other'")
            
            # 重新统计
            new_class_counts = Counter(self.df[target_column])
            print(f"处理后类别分布:")
            for cls, count in sorted(new_class_counts.items()):
                print(f"  - {cls}: {count} 样本")
        else:
            print(f"未发现稀有类别（所有类别样本数 >= {min_count}）")
        
        return self.df
    
    def encode_categorical_features(self):
        """编码分类特征"""
        if self.df is None:
            print("请先加载和清洗数据！")
            return None
            
        categorical_columns = ['Co-material', 'F-base', 'code', 'Secondary Morphology']
        
        for col in categorical_columns:
            if col in self.df.columns:
                # 创建编码后的列名
                encoded_col_name = f"{col}_encoded"
                
                # 确保列为字符串类型，避免混合类型导致的编码错误
                self.df[col] = self.df[col].astype(str)
                
                # 创建标签编码器
                le = LabelEncoder()
                
                # 编码特征
                self.df[encoded_col_name] = le.fit_transform(self.df[col])
                
                # 保存编码器
                self.label_encoders[col] = le
                
                print(f"列 '{col}' 已编码为 '{encoded_col_name}'")
                print(f"  唯一值: {list(le.classes_)}")
        
        return self.df
    
    def prepare_features(self, target_column='Secondary Morphology'):
        """准备特征和目标变量"""
        if self.df is None:
            print("请先完成数据预处理！")
            return None, None
            
        # 选择数值特征（包括编码后的分类特征）
        numeric_features = [
            'r-time', 'r-temperature', 'c-time', 'c-temperature',
            'Co3+', 'urea', 'NH4F', 'SDS', 'load mass-mg/cm2'
        ]
        
        # 添加编码后的分类特征
        encoded_features = [col for col in self.df.columns if col.endswith('_encoded')]
        all_features = numeric_features + encoded_features
        
        # 检查特征是否存在
        available_features = [col for col in all_features if col in self.df.columns]
        print(f"可用特征: {available_features}")
        
        # 准备特征矩阵
        X = self.df[available_features].values
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 准备目标变量
        if f"{target_column}_encoded" in self.df.columns:
            y = self.df[f"{target_column}_encoded"].values
        elif target_column in self.df.columns:
            y = self.df[target_column].values
        else:
            print(f"目标列 '{target_column}' 不存在！")
            return None, None
        
        print(f"特征矩阵形状: {X_scaled.shape}")
        print(f"目标变量形状: {y.shape}")
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """分割训练集和测试集"""
        try:
            # 检查是否可以进行分层分割
            from collections import Counter
            class_counts = Counter(y)
            min_class_count = min(class_counts.values())
            
            if min_class_count < 2:
                print(f"警告: 最小类别样本数为 {min_class_count}，使用随机分割替代分层分割")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            else:
                print("使用分层分割...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            
            print(f"训练集大小: {X_train.shape[0]}")
            print(f"测试集大小: {X_test.shape[0]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"数据分割失败: {e}")
            return None, None, None, None
    
    def get_feature_names(self):
        """获取特征名称"""
        if self.df is None:
            return []
            
        numeric_features = [
            'r-time', 'r-temperature', 'c-time', 'c-temperature',
            'Co3+', 'urea', 'NH4F', 'SDS', 'load mass-mg/cm2'
        ]
        
        encoded_features = [col for col in self.df.columns if col.endswith('_encoded')]
        all_features = numeric_features + encoded_features
        
        return [col for col in all_features if col in self.df.columns]
    
    def get_data_summary(self):
        """获取数据摘要"""
        if self.df is None:
            return "数据未加载"
            
        summary = {
            "数据形状": self.df.shape,
            "列数": len(self.df.columns),
            "行数": len(self.df),
            "数值列": len(self.df.select_dtypes(include=[np.number]).columns),
            "分类列": len(self.df.select_dtypes(include=['object']).columns),
            "缺失值总数": self.df.isnull().sum().sum()
        }
        
        return summary

def main():
    """主函数 - 演示数据预处理流程"""
    # 导入配置
    import sys
    import os
    
    # 获取项目根目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 添加项目根目录到Python路径
    sys.path.append(project_root)
    
    from config import DATASET_PATHS
    
    # 修正数据文件路径
    data_path = os.path.join(project_root, DATASET_PATHS['main_dataset'])
    
    # 创建预处理器
    preprocessor = DataPreprocessor(data_path)
    
    # 加载数据
    df = preprocessor.load_data()
    
    # 清洗数据
    df = preprocessor.clean_data()
    
    # 编码分类特征
    df = preprocessor.encode_categorical_features()
    
    # 准备特征
    X, y = preprocessor.prepare_features()
    
    if X is not None and y is not None:
        # 分割数据
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        # 打印数据摘要
        print("\n数据摘要:")
        summary = preprocessor.get_data_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n数据预处理完成！")
        return X_train, X_test, y_train, y_test, preprocessor
    
    return None

if __name__ == "__main__":
    main()
