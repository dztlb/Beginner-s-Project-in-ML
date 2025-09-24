"""
SHAP分析模块
使用SHAP (SHapley Additive exPlanations) 进行模型解释性分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
from font_config import setup_chinese_fonts
setup_chinese_fonts()
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SHAPAnalyzer:
    """SHAP分析器类"""
    
    def __init__(self, X, y, feature_names, task_type='classification'):
        """
        初始化SHAP分析器
        
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
        self.explainers = {}
        self.shap_values = {}
        
    def create_explainer(self, model, model_name):
        """
        为指定模型创建SHAP解释器
        
        Parameters:
        -----------
        model : object
            训练好的模型
        model_name : str
            模型名称
        """
        print(f"为 {model_name} 创建SHAP解释器...")
        
        try:
            # 根据模型类型创建相应的解释器
            if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
                explainer = shap.TreeExplainer(model)
            elif isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
                explainer = shap.TreeExplainer(model)
            else:
                # 对于其他类型的模型，使用KernelExplainer
                explainer = shap.KernelExplainer(model.predict, self.X[:100])  # 使用前100个样本
            
            self.explainers[model_name] = explainer
            print(f"  {model_name} 解释器创建成功")
            
        except Exception as e:
            print(f"  {model_name} 解释器创建失败: {e}")
            return None
        
        return explainer
    
    def calculate_shap_values(self, model_name, sample_size=None):
        """
        计算SHAP值
        
        Parameters:
        -----------
        model_name : str
            模型名称
        sample_size : int, optional
            样本大小，如果为None则使用全部数据
        """
        if model_name not in self.explainers:
            print(f"模型 {model_name} 的解释器不存在！")
            return None
        
        explainer = self.explainers[model_name]
        
        # 选择样本
        if sample_size and sample_size < len(self.X):
            indices = np.random.choice(len(self.X), sample_size, replace=False)
            X_sample = self.X[indices]
        else:
            X_sample = self.X
            indices = np.arange(len(self.X))
        
        print(f"计算 {model_name} 的SHAP值，样本数: {len(X_sample)}")
        
        try:
            # 计算SHAP值
            if isinstance(explainer, shap.TreeExplainer):
                shap_values = explainer.shap_values(X_sample)
                
                # 处理多分类情况
                if isinstance(shap_values, list):
                    # 多分类情况，选择第一个类别或计算平均值
                    if len(shap_values) > 0:
                        # 对于多分类，我们选择第一个类别的SHAP值
                        shap_values = shap_values[0]
                        print(f"  多分类模型，使用第一个类别的SHAP值，形状: {shap_values.shape}")
                    else:
                        print("  SHAP值列表为空！")
                        return None
                elif len(shap_values.shape) == 3:
                    # 形状为 (样本数, 特征数, 类别数) 的情况
                    print(f"  多分类模型，SHAP值形状: {shap_values.shape}")
                    # 选择第一个类别或计算所有类别的平均值
                    shap_values = shap_values[:, :, 0]  # 选择第一个类别
                    print(f"  选择第一个类别的SHAP值，新形状: {shap_values.shape}")
                else:
                    print(f"  单分类模型，SHAP值形状: {shap_values.shape}")
            else:
                shap_values = explainer.shap_values(X_sample)
                print(f"  非树模型，SHAP值形状: {shap_values.shape}")
            
            # 验证SHAP值与样本数据的形状匹配
            if len(shap_values) != len(X_sample):
                print(f"  SHAP值样本数 ({len(shap_values)}) 与数据样本数 ({len(X_sample)}) 不匹配！")
                return None
            
            if len(shap_values.shape) > 1 and shap_values.shape[1] != len(self.feature_names):
                print(f"  SHAP值特征数 ({shap_values.shape[1]}) 与特征名称数 ({len(self.feature_names)}) 不匹配！")
                return None
            
            # 保存SHAP值
            self.shap_values[model_name] = {
                'shap_values': shap_values,
                'X_sample': X_sample,
                'indices': indices
            }
            
            print(f"  {model_name} SHAP值计算完成，最终形状: {shap_values.shape}")
            return shap_values
            
        except Exception as e:
            print(f"  {model_name} SHAP值计算失败: {e}")
            return None
    
    def plot_summary_plot(self, model_name, save_path=None):
        """
        绘制SHAP摘要图
        
        Parameters:
        -----------
        model_name : str
            模型名称
        save_path : str, optional
            图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"模型 {model_name} 的SHAP值不存在！")
            return
        
        print(f"绘制 {model_name} 的SHAP摘要图...")
        
        try:
            explainer = self.explainers[model_name]
            shap_data = self.shap_values[model_name]
            
            # 获取SHAP值和样本数据
            shap_values = shap_data['shap_values']
            X_sample = shap_data['X_sample']
            
            # 确保数据形状匹配
            if len(shap_values) != len(X_sample):
                print(f"SHAP值样本数 ({len(shap_values)}) 与数据样本数 ({len(X_sample)}) 不匹配！")
                return
            
            # 创建摘要图
            plt.figure(figsize=(10, 8))
            
            try:
                # 使用SHAP的summary_plot
                shap.summary_plot(
                    shap_values, 
                    X_sample,
                    feature_names=self.feature_names,
                    show=False
                )
            except Exception as e:
                print(f"SHAP summary_plot失败，使用手动方法: {e}")
                # 手动创建摘要图
                mean_abs_shap = np.abs(shap_values).mean(0)
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': mean_abs_shap
                }).sort_values('Importance', ascending=True)
                
                plt.barh(range(len(feature_importance)), feature_importance['Importance'])
                plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
                plt.xlabel('Mean |SHAP value|')
                plt.title(f'{model_name} 特征重要性 (SHAP)')
            
            plt.title(f'{model_name} SHAP特征重要性摘要')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP摘要图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"SHAP摘要图绘制失败: {e}")
    
    def plot_bar_plot(self, model_name, save_path=None):
        """
        绘制SHAP条形图
        
        Parameters:
        -----------
        model_name : str
            模型名称
        save_path : str, optional
            图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"模型 {model_name} 的SHAP值不存在！")
            return
        
        print(f"绘制 {model_name} 的SHAP条形图...")
        
        try:
            shap_data = self.shap_values[model_name]
            
            # 获取SHAP值
            shap_values = shap_data['shap_values']
            
            # 创建条形图
            plt.figure(figsize=(10, 8))
            
            try:
                # 尝试使用SHAP的summary_plot创建条形图
                shap.summary_plot(
                    shap_values, 
                    shap_data['X_sample'],
                    feature_names=self.feature_names,
                    plot_type="bar",
                    show=False
                )
            except Exception as e:
                print(f"SHAP条形图失败，使用手动方法: {e}")
                # 手动创建条形图
                mean_abs_shap = np.abs(shap_values).mean(0)
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': mean_abs_shap
                }).sort_values('Importance', ascending=True)
                
                plt.barh(range(len(feature_importance)), feature_importance['Importance'])
                plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
                plt.xlabel('Mean |SHAP value|')
                plt.title(f'{model_name} 特征重要性 (SHAP)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP条形图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"SHAP条形图绘制失败: {e}")
    
    def plot_waterfall_plot(self, model_name, sample_index=0, save_path=None):
        """
        绘制SHAP瀑布图
        
        Parameters:
        -----------
        model_name : str
            模型名称
        sample_index : int
            样本索引
        save_path : str, optional
            图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"模型 {model_name} 的SHAP值不存在！")
            return
        
        print(f"绘制 {model_name} 的SHAP瀑布图，样本索引: {sample_index}...")
        
        try:
            explainer = self.explainers[model_name]
            shap_data = self.shap_values[model_name]
            
            if sample_index >= len(shap_data['X_sample']):
                print(f"样本索引 {sample_index} 超出范围！")
                return
            
            # 获取SHAP值和样本数据
            shap_values = shap_data['shap_values']
            X_sample = shap_data['X_sample']
            
            # 创建瀑布图
            plt.figure(figsize=(12, 8))
            
            try:
                # 尝试使用新版本的SHAP API
                import shap.plots as shap_plots
                shap_plots.waterfall(
                    explainer.expected_value,
                    shap_values[sample_index],
                    X_sample[sample_index]
                )
            except (ImportError, AttributeError):
                try:
                    # 尝试使用旧版本的waterfall_plot
                    shap.waterfall_plot(
                        explainer.expected_value,
                        shap_values[sample_index],
                        X_sample[sample_index],
                        show=False
                    )
                except Exception as e:
                    print(f"SHAP waterfall_plot失败，使用force_plot: {e}")
                    try:
                        # 使用force_plot作为备选
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[sample_index],
                            X_sample[sample_index],
                            show=False
                        )
                    except Exception as e2:
                        print(f"SHAP force_plot也失败，使用手动方法: {e2}")
                        # 手动创建简单的瀑布图
                        feature_contributions = shap_values[sample_index]
                        feature_values = X_sample[sample_index]
                        
                        # 创建贡献条形图
                        contributions_df = pd.DataFrame({
                            'Feature': self.feature_names,
                            'Contribution': feature_contributions,
                            'Value': feature_values
                        }).sort_values('Contribution', key=abs, ascending=False)
                        
                        plt.bar(range(len(contributions_df)), contributions_df['Contribution'])
                        plt.xticks(range(len(contributions_df)), contributions_df['Feature'], rotation=45, ha='right')
                        plt.ylabel('SHAP Contribution')
                        plt.title(f'{model_name} SHAP贡献 - 样本 {sample_index}')
            
            plt.title(f'{model_name} SHAP瀑布图 - 样本 {sample_index}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP瀑布图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"SHAP瀑布图绘制失败: {e}")
    
    def plot_dependence_plot(self, model_name, feature_name, save_path=None):
        """
        绘制SHAP依赖图
        
        Parameters:
        -----------
        model_name : str
            模型名称
        feature_name : str
            特征名称
        save_path : str, optional
            图片保存路径
        """
        if model_name not in self.shap_values:
            print(f"模型 {model_name} 的SHAP值不存在！")
            return
        
        if feature_name not in self.feature_names:
            print(f"特征 {feature_name} 不存在！")
            return
        
        print(f"绘制 {model_name} 的SHAP依赖图，特征: {feature_name}...")
        
        try:
            shap_data = self.shap_values[model_name]
            
            # 获取特征索引
            feature_index = self.feature_names.index(feature_name)
            
            # 获取SHAP值和样本数据
            shap_values = shap_data['shap_values']
            X_sample = shap_data['X_sample']
            
            # 创建依赖图
            plt.figure(figsize=(10, 8))
            
            try:
                # 尝试使用SHAP的dependence_plot
                shap.dependence_plot(
                    feature_index,
                    shap_values,
                    X_sample,
                    feature_names=self.feature_names,
                    show=False
                )
            except Exception as e:
                print(f"SHAP dependence_plot失败，使用手动方法: {e}")
                # 手动创建依赖图
                feature_values = X_sample[:, feature_index]
                shap_values_for_feature = shap_values[:, feature_index]
                
                plt.scatter(feature_values, shap_values_for_feature, alpha=0.6)
                plt.xlabel(feature_name)
                plt.ylabel(f'SHAP value for {feature_name}')
                plt.title(f'{model_name} SHAP依赖图 - {feature_name}')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP依赖图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"SHAP依赖图绘制失败: {e}")
    
    def get_feature_importance_ranking(self, model_name):
        """
        获取特征重要性排序
        
        Parameters:
        -----------
        model_name : str
            模型名称
        
        Returns:
        --------
        DataFrame : 特征重要性排序
        """
        if model_name not in self.shap_values:
            print(f"模型 {model_name} 的SHAP值不存在！")
            return None
        
        try:
            shap_data = self.shap_values[model_name]
            
            # 获取SHAP值
            shap_values = shap_data['shap_values']
            
            # 验证SHAP值与特征名称的匹配
            if len(shap_values.shape) > 1 and shap_values.shape[1] != len(self.feature_names):
                print(f"SHAP值特征数 ({shap_values.shape[1]}) 与特征名称数 ({len(self.feature_names)}) 不匹配！")
                return None
            
            # 计算平均绝对SHAP值
            mean_abs_shap = np.abs(shap_values).mean(0)
            
            # 验证平均SHAP值的长度
            if len(mean_abs_shap) != len(self.feature_names):
                print(f"平均SHAP值长度 ({len(mean_abs_shap)}) 与特征名称数 ({len(self.feature_names)}) 不匹配！")
                return None
            
            # 创建特征重要性DataFrame
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Mean_|SHAP|': mean_abs_shap,
                'Rank': range(1, len(self.feature_names) + 1)
            })
            
            # 按重要性排序
            feature_importance = feature_importance.sort_values('Mean_|SHAP|', ascending=False)
            feature_importance['Rank'] = range(1, len(self.feature_names) + 1)
            
            return feature_importance
            
        except Exception as e:
            print(f"获取特征重要性排序失败: {e}")
            return None
    
    def save_shap_analysis(self, output_path):
        """
        保存SHAP分析结果
        
        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        if not self.shap_values:
            print("没有SHAP分析结果可保存！")
            return
        
        print(f"保存SHAP分析结果到: {output_path}")
        
        try:
            # 创建结果摘要
            results_summary = []
            
            for model_name in self.shap_values.keys():
                # 获取特征重要性
                feature_importance = self.get_feature_importance_ranking(model_name)
                
                if feature_importance is not None:
                    # 添加模型信息
                    feature_importance['Model'] = model_name
                    results_summary.append(feature_importance)
            
            # 合并所有结果
            if results_summary:
                all_results = pd.concat(results_summary, ignore_index=True)
                
                # 保存为Excel文件
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    all_results.to_excel(writer, sheet_name='All_Results', index=False)
                    
                    # 为每个模型创建单独的工作表
                    for model_name in self.shap_values.keys():
                        model_results = all_results[all_results['Model'] == model_name]
                        if not model_results.empty:
                            model_results.to_excel(writer, sheet_name=model_name, index=False)
                
                print(f"SHAP分析结果已保存到: {output_path}")
                return all_results
            
        except Exception as e:
            print(f"保存SHAP分析结果失败: {e}")
            return None

def main():
    """主函数 - 演示SHAP分析流程"""
    print("=" * 60)
    print("SHAP分析程序启动")
    print("=" * 60)
    
    try:
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
        
        print(f"数据文件路径: {data_path}")
        
        # 数据预处理
        print("\n1. 开始数据预处理...")
        preprocessor = DataPreprocessor(data_path)
        preprocessor.load_data()
        preprocessor.clean_data()
        preprocessor.encode_categorical_features()
        
        # 准备特征
        X, y = preprocessor.prepare_features()
        feature_names = preprocessor.get_feature_names()
        
        if X is not None and y is not None:
            print(f"\n2. 数据准备完成:")
            print(f"   - 特征矩阵形状: {X.shape}")
            print(f"   - 目标变量形状: {y.shape}")
            print(f"   - 特征数量: {len(feature_names)}")
            
            # 创建SHAP分析器
            print("\n3. 创建SHAP分析器...")
            analyzer = SHAPAnalyzer(X, y, feature_names, task_type='classification')
            
            # 训练一个简单的模型用于演示
            print("\n4. 训练随机森林模型...")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            print("   模型训练完成")
            
            # 创建解释器
            print("\n5. 创建SHAP解释器...")
            explainer = analyzer.create_explainer(model, 'RandomForest')
            
            if explainer is not None:
                # 计算SHAP值
                print("\n6. 计算SHAP值...")
                shap_values = analyzer.calculate_shap_values('RandomForest', sample_size=100)
                
                if shap_values is not None:
                    print("\n7. 开始生成SHAP可视化图表...")
                    
                    # 绘制各种SHAP图
                    print("   - 生成SHAP摘要图...")
                    analyzer.plot_summary_plot('RandomForest')
                    
                    print("   - 生成SHAP条形图...")
                    analyzer.plot_bar_plot('RandomForest')
                    
                    print("   - 生成SHAP瀑布图...")
                    analyzer.plot_waterfall_plot('RandomForest', sample_index=0)
                    
                    # 绘制前几个重要特征的依赖图
                    print("   - 生成特征依赖图...")
                    feature_importance = analyzer.get_feature_importance_ranking('RandomForest')
                    if feature_importance is not None:
                        top_features = feature_importance.head(3)['Feature'].tolist()
                        for feature in top_features:
                            print(f"     - 特征: {feature}")
                            analyzer.plot_dependence_plot('RandomForest', feature)
                    
                    # 保存结果
                    print("\n8. 保存SHAP分析结果...")
                    result_df = analyzer.save_shap_analysis('shap_analysis_results.xlsx')
                    
                    if result_df is not None:
                        print("\n" + "=" * 60)
                        print("SHAP分析完成！")
                        print("=" * 60)
                        print(f"结果已保存到: shap_analysis_results.xlsx")
                        print(f"分析的特征数量: {len(feature_names)}")
                        print(f"使用的样本数量: {len(shap_values)}")
                        
                        # 显示前5个最重要的特征
                        if feature_importance is not None:
                            print("\n前5个最重要的特征:")
                            for i, row in feature_importance.head(5).iterrows():
                                print(f"  {row['Rank']}. {row['Feature']}: {row['Mean_|SHAP|']:.6f}")
                        
                        return analyzer
                    else:
                        print("保存结果失败！")
                else:
                    print("SHAP值计算失败！")
            else:
                print("SHAP解释器创建失败！")
        else:
            print("数据准备失败！")
            
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return None

if __name__ == "__main__":
    main()
