"""
MgCo2O4电极材料机器学习项目主程序
整合数据预处理、特征选择、模型训练和SHAP分析等模块
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体支持
from src.font_config import setup_chinese_fonts
setup_chinese_fonts()

# 添加src目录到Python路径
sys.path.append('src')

# 导入配置
from config import DATASET_PATHS, MODEL_CONFIG, FEATURE_SELECTION_CONFIG, HYPERPARAMETER_CONFIG, OUTPUT_DIRS

def main():
    """主函数 - 执行完整的机器学习流程"""
    print("=" * 60)
    print("MgCo2O4电极材料机器学习项目")
    print("=" * 60)
    
    try:
        # 1. 数据预处理
        print("\n1. 开始数据预处理...")
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(DATASET_PATHS['main_dataset'])
        df = preprocessor.load_data()
        
        if df is None:
            print("数据加载失败！请检查文件路径。")
            return
        
        df = preprocessor.clean_data()
        
        # 处理稀有类别
        df = preprocessor.handle_rare_classes(target_column='Secondary Morphology', min_count=2)
        
        df = preprocessor.encode_categorical_features()
        
        # 准备特征
        X, y = preprocessor.prepare_features()
        feature_names = preprocessor.get_feature_names()
        
        if X is None or y is None:
            print("特征准备失败！")
            return
        
        print(f"数据预处理完成！特征矩阵形状: {X.shape}")
        
        # 2. 特征选择
        print("\n2. 开始特征选择...")
        from feature_selection import FeatureSelector
        
        selector = FeatureSelector(X, y, feature_names, task_type='classification')
        rfe_results = selector.perform_rfe(n_features_to_select=FEATURE_SELECTION_CONFIG['n_features_to_select'])
        
        if not rfe_results:
            print("特征选择失败！")
            return
        
        # 评估特征选择结果
        evaluation_results = selector.evaluate_feature_selection()
        
        # 绘制特征选择结果
        selector.plot_feature_ranking()
        selector.plot_performance_comparison(evaluation_results)
        
        # 获取最佳特征集
        best_model, best_features = selector.get_best_feature_set()
        
        # 检查特征选择是否成功
        if best_model is None or best_features is None:
            print("特征选择失败！无法获取最佳特征集。")
            return False
        
        # 保存特征选择结果
        selector.save_results('results/feature_selection_results.xlsx')
        
        print("特征选择完成！")
        
        # 3. 模型训练
        print("\n3. 开始模型训练...")
        from model_training import ModelTrainer
        
        trainer = ModelTrainer(X, y, feature_names, task_type='classification')
        training_results = trainer.train_models()
        
        if not training_results:
            print("模型训练失败！")
            return
        
        # 构建Stacking模型（若失败则跳过，不影响后续流程）
        stacking_model = trainer.build_stacking_model()
        if stacking_model is None:
            print("Stacking模型构建失败，继续执行其他步骤...")
        
        # 超参数调优示例
        print("\n4. 开始超参数调优...")
        trainer.hyperparameter_tuning('RF', HYPERPARAMETER_CONFIG['RF'])
        
        # 绘制性能对比
        trainer.plot_performance_comparison()
        
        # 绘制混淆矩阵
        if trainer.best_model_name:
            trainer.plot_confusion_matrix(trainer.best_model_name)
        
        # 保存模型和结果
        trainer.save_models()
        trainer.save_results('results/training_results.xlsx')
        
        print("模型训练完成！")
        
        # 5. SHAP分析
        print("\n5. 开始SHAP分析...")
        from shap_analysis import SHAPAnalyzer
        
        analyzer = SHAPAnalyzer(X, y, feature_names, task_type='classification')
        
        # 为最佳模型创建SHAP解释器
        if trainer.best_model_name in trainer.trained_models:
            best_model = trainer.trained_models[trainer.best_model_name]
            explainer = analyzer.create_explainer(best_model, trainer.best_model_name)
            
            if explainer:
                # 计算SHAP值
                shap_values = analyzer.calculate_shap_values(
                    trainer.best_model_name, sample_size=100
                )
                
                if shap_values is not None:
                    # 绘制SHAP图
                    analyzer.plot_summary_plot(trainer.best_model_name)
                    analyzer.plot_bar_plot(trainer.best_model_name)
                    analyzer.plot_waterfall_plot(trainer.best_model_name, sample_index=0)
                    
                    # 绘制重要特征的依赖图
                    feature_importance = analyzer.get_feature_importance_ranking(trainer.best_model_name)
                    if feature_importance is not None:
                        top_features = feature_importance.head(3)['Feature'].tolist()
                        for feature in top_features:
                            analyzer.plot_dependence_plot(trainer.best_model_name, feature)
                    
                    # 保存SHAP分析结果
                    analyzer.save_shap_analysis('results/shap_analysis_results.xlsx')
                    
                    print("SHAP分析完成！")
        
        # 6. 结果总结
        print("\n" + "=" * 60)
        print("项目执行完成！结果总结：")
        print("=" * 60)
        
        # 数据预处理总结
        print("\n数据预处理结果：")
        summary = preprocessor.get_data_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # 特征选择总结
        print(f"\n特征选择结果：")
        if best_model is not None and best_features is not None:
            print(f"  最佳模型: {best_model}")
            print(f"  选择特征数: {len(best_features)}")
            print(f"  选择特征: {best_features}")
        else:
            print("  特征选择失败或未完成")
        
        # 模型训练总结
        print(f"\n模型训练结果：")
        model_summary = trainer.get_model_summary()
        for key, value in model_summary.items():
            print(f"  {key}: {value}")
        
        # 文件输出总结
        print(f"\n生成的文件：")
        output_files = [
            'results/feature_selection_results.xlsx',
            'results/training_results.xlsx',
            'results/shap_analysis_results.xlsx'
        ]
        
        for file in output_files:
            if os.path.exists(file):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (未生成)")
        
        if os.path.exists('models'):
            print(f"  ✓ models/ 目录 (包含训练好的模型)")
        
        print(f"\n项目执行成功！所有模块运行完成。")
        
    except Exception as e:
        print(f"\n项目执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_individual_modules():
    """运行单独的模块（用于调试）"""
    print("选择要运行的模块：")
    print("1. 数据预处理")
    print("2. 特征选择")
    print("3. 模型训练")
    print("4. SHAP分析")
    print("5. 运行完整流程")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    if choice == '1':
        print("\n运行数据预处理模块...")
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(DATASET_PATHS['main_dataset'])
        preprocessor.load_data()
        preprocessor.clean_data()
        preprocessor.encode_categorical_features()
        X, y = preprocessor.prepare_features()
        print("数据预处理完成！")
        
    elif choice == '2':
        print("\n运行特征选择模块...")
        from feature_selection import main as feature_selection_main
        feature_selection_main()
        
    elif choice == '3':
        print("\n运行模型训练模块...")
        from model_training import main as model_training_main
        model_training_main()
        
    elif choice == '4':
        print("\n运行SHAP分析模块...")
        from shap_analysis import main as shap_analysis_main
        shap_analysis_main()
        
    elif choice == '5':
        print("\n运行完整流程...")
        main()
        
    else:
        print("无效选择！")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_individual_modules()
    else:
        # 运行完整流程
        success = main()
        if success:
            print("\n项目执行成功！")
        else:
            print("\n项目执行失败！")
            sys.exit(1)
