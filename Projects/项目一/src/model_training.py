"""
模型训练模块
训练多种机器学习模型并进行性能评估
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix, classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
from font_config import setup_chinese_fonts
setup_chinese_fonts()
import joblib
import os

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, X, y, feature_names, task_type='classification'):
        """
        初始化模型训练器
        
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
        
        # 根据任务类型选择模型
        if task_type == 'classification':
            self.models = {
                'RF': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGB': xgb.XGBClassifier(random_state=42),
                'DT': DecisionTreeClassifier(random_state=42)
            }
        else:
            self.models = {
                'RF': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGB': xgb.XGBRegressor(random_state=42),
                'DT': DecisionTreeRegressor(random_state=42)
            }
        
        self.trained_models = {}
        self.training_results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, cv_folds=5):
        """
        训练所有模型
        
        Parameters:
        -----------
        cv_folds : int
            交叉验证折数
        """
        print("开始训练模型...")
        
        # 分割训练集和测试集
        try:
            if self.task_type == 'classification':
                # 检查是否可以进行分层分割
                from collections import Counter
                class_counts = Counter(self.y)
                min_class_count = min(class_counts.values())
                
                if min_class_count < 2:
                    print(f"警告: 最小类别样本数为 {min_class_count}，使用随机分割替代分层分割")
                    X_train, X_test, y_train, y_test = train_test_split(
                        self.X, self.y, test_size=0.2, random_state=42
                    )
                else:
                    print("使用分层分割...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=42
                )
        except Exception as e:
            print(f"分层分割失败，使用随机分割: {e}")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 训练每个模型
        for name, model in self.models.items():
            print(f"\n训练 {name} 模型...")
            
            try:
                # 交叉验证
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv_folds,
                    scoring='accuracy' if self.task_type == 'classification' else 'r2'
                )
                
                # 训练最终模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 计算性能指标
                if self.task_type == 'classification':
                    metrics = self._calculate_classification_metrics(y_test, y_pred)
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                
                # 保存结果
                self.trained_models[name] = model
                self.training_results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores,
                    'test_metrics': metrics,
                    'y_pred': y_pred
                }
                
                print(f"  {name} 训练完成")
                print(f"  交叉验证性能: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  {name} 训练失败: {e}")
        
        # 找到最佳模型
        self._find_best_model()
        
        return self.training_results
    
    def _calculate_classification_metrics(self, y_true, y_pred):
        """计算分类任务性能指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def _calculate_regression_metrics(self, y_true, y_pred):
        """计算回归任务性能指标"""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }
    
    def _find_best_model(self):
        """找到最佳模型"""
        if not self.training_results:
            return
        
        # 根据交叉验证性能选择最佳模型
        best_score = -np.inf
        best_model_name = None
        
        for name, results in self.training_results.items():
            # 优先使用交叉验证性能，如果没有则使用测试集性能
            if 'cv_mean' in results:
                score = results['cv_mean']
            elif 'test_metrics' in results and 'accuracy' in results['test_metrics']:
                score = results['test_metrics']['accuracy']
            elif 'test_metrics' in results and 'r2' in results['test_metrics']:
                score = results['test_metrics']['r2']
            else:
                continue  # 跳过没有性能指标的模型
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        if best_model_name:
            self.best_model_name = best_model_name
            self.best_model = self.trained_models[best_model_name]
            print(f"\n最佳模型: {self.best_model_name}")
            print(f"最佳性能: {best_score:.4f}")
        else:
            print("\n未找到有效的模型性能指标！")
    
    def build_stacking_model(self, cv_folds=5):
        """
        构建Stacking融合模型
        
        Parameters:
        -----------
        cv_folds : int
            交叉验证折数
        """
        print("\n构建Stacking融合模型...")
        
        if not self.trained_models:
            print("请先训练基模型！")
            return None
        
        try:
            # 准备基模型（排除在CV中对类别缺失不够健壮的XGB）
            base_models = [
                (name, model) for name, model in self.trained_models.items() if name != 'XGB'
            ]
            if len(base_models) < 2:
                print("可用于Stacking的基模型不足，跳过Stacking（需要至少2个模型）")
                return None
            
            # 选择元学习器
            if self.task_type == 'classification':
                meta_learner = LogisticRegression(random_state=42)
            else:
                meta_learner = LinearRegression()
            
            # 构建Stacking模型 - 使用较少折数并进行分层以降低类别缺失风险
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=min(3, len(np.unique(self.y))), shuffle=True, random_state=42)
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=skf  # 使用分层K折
            )
            
            # 训练Stacking模型
            stacking_model.fit(self.X_train, self.y_train)
            
            # 预测
            y_pred_stacking = stacking_model.predict(self.X_test)
            
            # 计算性能指标
            if self.task_type == 'classification':
                metrics = self._calculate_classification_metrics(self.y_test, y_pred_stacking)
            else:
                metrics = self._calculate_regression_metrics(self.y_test, y_pred_stacking)
            
            # 交叉验证
            cv_scores = cross_val_score(
                stacking_model, self.X, self.y, cv=cv_folds,
                scoring='accuracy' if self.task_type == 'classification' else 'r2'
            )
            
            # 保存结果
            self.trained_models['Stacking'] = stacking_model
            self.training_results['Stacking'] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'test_metrics': metrics,
                'y_pred': y_pred_stacking
            }
            
            print("Stacking模型训练完成！")
            print(f"交叉验证性能: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return stacking_model
            
        except Exception as e:
            print(f"Stacking模型构建失败: {e}")
            return None
    
    def hyperparameter_tuning(self, model_name, param_grid, cv_folds=5):
        """
        超参数调优
        
        Parameters:
        -----------
        model_name : str
            要调优的模型名称
        param_grid : dict
            参数网格
        cv_folds : int
            交叉验证折数
        """
        if model_name not in self.models:
            print(f"模型 {model_name} 不存在！")
            return None
        
        print(f"\n对 {model_name} 进行超参数调优...")
        
        try:
            # 创建网格搜索对象
            grid_search = GridSearchCV(
                self.models[model_name],
                param_grid,
                cv=cv_folds,
                scoring='accuracy' if self.task_type == 'classification' else 'r2',
                n_jobs=-1,
                verbose=1
            )
            
            # 执行网格搜索
            grid_search.fit(self.X_train, self.y_train)
            
            # 获取最佳参数和性能
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"最佳参数: {best_params}")
            print(f"最佳性能: {best_score:.4f}")
            
            # 使用最佳参数更新模型
            best_model = grid_search.best_estimator_
            self.trained_models[f"{model_name}_Tuned"] = best_model
            
            # 在测试集上评估
            y_pred = best_model.predict(self.X_test)
            
            if self.task_type == 'classification':
                metrics = self._calculate_classification_metrics(self.y_test, y_pred)
            else:
                metrics = self._calculate_regression_metrics(self.y_test, y_pred)
            
            # 保存调优结果
            self.training_results[f"{model_name}_Tuned"] = {
                'best_params': best_params,
                'best_score': best_score,
                'test_metrics': metrics,
                'y_pred': y_pred
            }
            
            return best_model, best_params
            
        except Exception as e:
            print(f"超参数调优失败: {e}")
            return None, None
    
    def plot_performance_comparison(self, save_path=None):
        """绘制模型性能对比图"""
        if not self.training_results:
            print("请先训练模型！")
            return
        
        # 准备数据 - 只选择成功训练的模型
        models = []
        cv_means = []
        cv_stds = []
        
        for model_name, results in self.training_results.items():
            if 'cv_mean' in results and 'cv_std' in results:
                models.append(model_name)
                cv_means.append(results['cv_mean'])
                cv_stds.append(results['cv_std'])
        
        if not models:
            print("没有成功训练的模型可以绘制！")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 交叉验证性能对比
        x = np.arange(len(models))
        bars = ax1.bar(x, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
        ax1.set_xlabel('模型')
        ax1.set_ylabel('性能指标')
        ax1.set_title('交叉验证性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 测试集性能对比
        if self.task_type == 'classification':
            test_metrics = [self.training_results[model]['test_metrics']['accuracy'] for model in models]
            metric_name = 'Accuracy'
        else:
            test_metrics = [self.training_results[model]['test_metrics']['r2'] for model in models]
            metric_name = 'R²'
        
        bars2 = ax2.bar(x, test_metrics, alpha=0.7, color='orange')
        ax2.set_xlabel('模型')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'测试集{metric_name}对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        
        # 添加数值标签
        for i, (bar, metric) in enumerate(zip(bars2, test_metrics)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{metric:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """绘制混淆矩阵（仅分类任务）"""
        if self.task_type != 'classification':
            print("混淆矩阵仅适用于分类任务！")
            return
        
        if model_name not in self.training_results:
            print(f"模型 {model_name} 不存在！")
            return
        
        # 计算混淆矩阵
        y_pred = self.training_results[model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def save_models(self, output_dir='models'):
        """保存训练好的模型"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n保存模型到 {output_dir} 目录...")
        
        for name, model in self.trained_models.items():
            try:
                model_path = os.path.join(output_dir, f"{name}_model.pkl")
                joblib.dump(model, model_path)
                print(f"  {name} 模型已保存到: {model_path}")
            except Exception as e:
                print(f"  {name} 模型保存失败: {e}")
    
    def save_results(self, output_path):
        """保存训练结果"""
        if not self.training_results:
            print("请先训练模型！")
            return
        
        # 创建结果摘要
        results_summary = []
        
        for name, results in self.training_results.items():
            summary = {'Model': name}
            
            # 添加交叉验证指标（如果存在）
            if 'cv_mean' in results:
                summary['CV_Mean'] = results['cv_mean']
            else:
                summary['CV_Mean'] = 'N/A'
                
            if 'cv_std' in results:
                summary['CV_Std'] = results['cv_std']
            else:
                summary['CV_Std'] = 'N/A'
            
            # 添加测试集指标
            if 'test_metrics' in results:
                for metric, value in results['test_metrics'].items():
                    summary[f'Test_{metric}'] = value
            
            # 添加其他信息（如超参数调优结果）
            if 'best_params' in results:
                summary['Best_Params'] = str(results['best_params'])
            if 'best_score' in results:
                summary['Best_Score'] = results['best_score']
            
            results_summary.append(summary)
        
        # 保存为Excel文件
        df_summary = pd.DataFrame(results_summary)
        df_summary.to_excel(output_path, index=False)
        
        print(f"训练结果已保存到: {output_path}")
        
        return df_summary
    
    def get_model_summary(self):
        """获取模型训练摘要"""
        if not self.training_results:
            return "模型未训练"
        
        summary = {
            "训练模型数量": len(self.trained_models),
            "最佳模型": self.best_model_name,
            "模型列表": list(self.trained_models.keys())
        }
        
        return summary

def main():
    """主函数 - 演示模型训练流程"""
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
        # 创建模型训练器
        trainer = ModelTrainer(X, y, feature_names, task_type='classification')
        
        # 训练模型
        training_results = trainer.train_models()
        
        # 构建Stacking模型
        stacking_model = trainer.build_stacking_model()
        
        # 超参数调优示例
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        trainer.hyperparameter_tuning('RF', param_grid_rf)
        
        # 绘制性能对比
        trainer.plot_performance_comparison()
        
        # 绘制混淆矩阵
        if trainer.best_model_name:
            trainer.plot_confusion_matrix(trainer.best_model_name)
        
        # 保存模型和结果
        trainer.save_models()
        trainer.save_results('training_results.xlsx')
        
        # 打印摘要
        print("\n模型训练摘要:")
        summary = trainer.get_model_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n模型训练完成！")
        return trainer, training_results
    
    return None

if __name__ == "__main__":
    main()
