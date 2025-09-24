"""
特征选择模块
使用递归特征消除(RFE)方法选择最优特征子集
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
from font_config import setup_chinese_fonts
setup_chinese_fonts()

class FeatureSelector:
    """特征选择器类"""
    
    def __init__(self, X, y, feature_names, task_type='classification'):
        """
        初始化特征选择器
        
        Parameters:
        -----------
        X : array-like
            特征矩阵
        y : array-like
            目标变量
        feature_names : list
            特征名称列表
        task_type : str
            任务类型 ('classification' 或 'regression')
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.task_type = task_type
        
        # 根据任务类型选择基模型
        if task_type == 'classification':
            self.base_models = {
                'RF': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGB': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'DT': DecisionTreeClassifier(random_state=42)
            }
        else:
            self.base_models = {
                'RF': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGB': xgb.XGBRegressor(random_state=42),
                'DT': DecisionTreeRegressor(random_state=42)
            }
        
        self.rfe_results = {}
        self.selected_features = {}
        
    def perform_rfe(self, n_features_to_select=8):
        """
        执行递归特征消除
        
        Parameters:
        -----------
        n_features_to_select : int
            要选择的特征数量
        """
        print(f"开始RFE特征选择，目标特征数: {n_features_to_select}")
        
        for name, model in self.base_models.items():
            print(f"\n使用 {name} 作为RFE基模型...")
            
            try:
                # 创建RFE对象
                rfe = RFE(
                    estimator=model,
                    n_features_to_select=n_features_to_select,
                    step=1
                )
                
                # 执行RFE
                X_rfe = rfe.fit_transform(self.X, self.y)
                
                # 获取选择的特征
                selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if rfe.support_[i]]
                
                # 保存结果
                self.rfe_results[name] = {
                    'rfe': rfe,
                    'X_selected': X_rfe,
                    'selected_features': selected_features,
                    'feature_ranking': rfe.ranking_,
                    'feature_support': rfe.support_
                }
                
                self.selected_features[name] = selected_features
                
                print(f"  {name} 选择的特征: {selected_features}")
                print(f"  特征重要性排序: {rfe.ranking_}")
                
            except Exception as e:
                print(f"  {name} RFE执行失败: {e}")
        
        return self.rfe_results
    
    def evaluate_feature_selection(self, cv_folds=5):
        """
        评估特征选择结果
        
        Parameters:
        -----------
        cv_folds : int
            交叉验证折数
        """
        print("\n评估特征选择结果...")
        
        evaluation_results = {}
        
        for name, result in self.rfe_results.items():
            print(f"\n评估 {name} 特征选择结果...")
            
            try:
                # 原始特征集性能
                if name == 'XGB':
                    # 对于XGBoost，使用StratifiedKFold确保每个折都包含所有类别
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    original_scores = cross_val_score(
                        self.base_models[name], 
                        self.X, 
                        self.y, 
                        cv=skf,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2'
                    )
                    
                    selected_scores = cross_val_score(
                        self.base_models[name], 
                        result['X_selected'], 
                        self.y, 
                        cv=skf,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2'
                    )
                else:
                    # 其他模型使用普通交叉验证
                    original_scores = cross_val_score(
                        self.base_models[name], 
                        self.X, 
                        self.y, 
                        cv=cv_folds,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2'
                    )
                    
                    selected_scores = cross_val_score(
                        self.base_models[name], 
                        result['X_selected'], 
                        self.y, 
                        cv=cv_folds,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2'
                    )
                
                evaluation_results[name] = {
                    'original_mean': original_scores.mean(),
                    'original_std': original_scores.std(),
                    'selected_mean': selected_scores.mean(),
                    'selected_std': selected_scores.std(),
                    'improvement': selected_scores.mean() - original_scores.mean()
                }
                
                print(f"  原始特征集性能: {original_scores.mean():.4f} ± {original_scores.std():.4f}")
                print(f"  选择后特征集性能: {selected_scores.mean():.4f} ± {selected_scores.std():.4f}")
                print(f"  性能提升: {evaluation_results[name]['improvement']:.4f}")
                
            except Exception as e:
                print(f"  评估失败: {e}")
                evaluation_results[name] = {
                    'original_mean': 0.0,
                    'original_std': 0.0,
                    'selected_mean': 0.0,
                    'selected_std': 0.0,
                    'improvement': 0.0
                }
        
        # 设置实例属性
        self.evaluation_results = evaluation_results
        
        return evaluation_results
    
    def plot_feature_ranking(self, save_path=None):
        """
        绘制特征重要性排序图
        
        Parameters:
        -----------
        save_path : str, optional
            图片保存路径
        """
        if not self.rfe_results:
            print("请先执行RFE特征选择！")
            return
        
        n_models = len(self.rfe_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(self.rfe_results.items()):
            # 创建特征重要性DataFrame
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Ranking': result['feature_ranking'],
                'Selected': result['feature_support']
            })
            
            # 按重要性排序
            feature_importance_df = feature_importance_df.sort_values('Ranking')
            
            # 绘制条形图
            colors = ['red' if not selected else 'green' for selected in feature_importance_df['Selected']]
            axes[i].barh(range(len(feature_importance_df)), feature_importance_df['Ranking'], color=colors)
            axes[i].set_yticks(range(len(feature_importance_df)))
            axes[i].set_yticklabels(feature_importance_df['Feature'])
            axes[i].set_xlabel('Feature Ranking')
            axes[i].set_title(f'{name} Feature Ranking')
            axes[i].invert_yaxis()
            
            # 添加选择标记
            for j, selected in enumerate(feature_importance_df['Selected']):
                if selected:
                    axes[i].text(0.5, j, '✓', ha='center', va='center', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征排序图已保存到: {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, evaluation_results, save_path=None):
        """
        绘制特征选择前后性能对比图
        
        Parameters:
        -----------
        evaluation_results : dict
            评估结果字典
        save_path : str, optional
            图片保存路径
        """
        if not evaluation_results:
            print("请先评估特征选择结果！")
            return
        
        models = list(evaluation_results.keys())
        original_means = [evaluation_results[model]['original_mean'] for model in models]
        selected_means = [evaluation_results[model]['selected_mean'] for model in models]
        original_stds = [evaluation_results[model]['original_std'] for model in models]
        selected_stds = [evaluation_results[model]['selected_std'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制原始特征集性能
        rects1 = ax.bar(x - width/2, original_means, width, 
                       label='原始特征集', yerr=original_stds, capsize=5)
        
        # 绘制选择后特征集性能
        rects2 = ax.bar(x + width/2, selected_means, width, 
                       label='选择后特征集', yerr=selected_stds, capsize=5)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('性能指标')
        ax.set_title('特征选择前后性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {save_path}")
        
        plt.show()
    
    def get_best_feature_set(self, metric='improvement'):
        """
        获取最佳特征集
        
        Parameters:
        -----------
        metric : str
            选择标准 ('improvement', 'selected_mean', 'original_mean')
        
        Returns:
        --------
        tuple : (最佳模型名, 最佳特征集)
        """
        if not hasattr(self, 'evaluation_results'):
            print("请先评估特征选择结果！")
            return None, None
        
        # 找到最佳模型
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x][metric])
        
        best_features = self.selected_features[best_model]
        
        print(f"最佳模型: {best_model}")
        print(f"最佳特征集: {best_features}")
        
        return best_model, best_features
    
    def save_results(self, output_path):
        """
        保存特征选择结果
        
        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        if not self.rfe_results:
            print("请先执行RFE特征选择！")
            return
        
        # 创建结果摘要
        results_summary = []
        
        for name, result in self.rfe_results.items():
            summary = {
                'Model': name,
                'Selected_Features': ', '.join(result['selected_features']),
                'Feature_Count': len(result['selected_features']),
                'Feature_Names': result['selected_features']
            }
            results_summary.append(summary)
        
        # 保存为Excel文件
        df_summary = pd.DataFrame(results_summary)
        df_summary.to_excel(output_path, index=False)
        
        print(f"特征选择结果已保存到: {output_path}")
        
        return df_summary

def main():
    """主函数 - 演示特征选择流程"""
    # 导入数据预处理模块
    from data_preprocessing import DataPreprocessor
    
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
    
    # 数据预处理
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.encode_categorical_features()
    
    # 准备特征
    X, y = preprocessor.prepare_features()
    feature_names = preprocessor.get_feature_names()
    
    if X is not None and y is not None:
        # 创建特征选择器
        selector = FeatureSelector(X, y, feature_names, task_type='classification')
        
        # 执行RFE
        rfe_results = selector.perform_rfe(n_features_to_select=8)
        
        # 评估结果
        evaluation_results = selector.evaluate_feature_selection()
        
        # 绘制结果
        selector.plot_feature_ranking()
        selector.plot_performance_comparison(evaluation_results)
        
        # 获取最佳特征集
        best_model, best_features = selector.get_best_feature_set()
        
        # 保存结果
        selector.save_results('feature_selection_results.xlsx')
        
        print("\n特征选择完成！")
        return selector, rfe_results, evaluation_results
    
    return None

if __name__ == "__main__":
    main()
