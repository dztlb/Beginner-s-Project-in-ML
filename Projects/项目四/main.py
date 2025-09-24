#!/usr/bin/env python3
"""
雷达回波预测项目 - 主入口脚本
"""

import sys
import os



def main():
    """主函数"""
    while True:
        print("\n=== 雷达回波预测系统 ===")
        print("1. 开始训练模型")
        print("2. 数据预处理")
        print("3. 数据集分割")
        print("4. 模型测试")
        print("5. 退出")
        
        choice = input("请选择操作 (1-5): ").strip()
        
        if choice == "1":
            print("\n开始训练...")
            try:
                from config import TRAINING_CONFIG
                print(f"使用模型类型: {TRAINING_CONFIG['model_type']}")
                from train import main as train_main
                train_main()
            except Exception as e:
                print(f"训练过程中出现错误: {e}")
        elif choice == "2":
            print("\n开始数据预处理...")
            try:
                # 简单的数据预处理功能
                from config import DATA_CONFIG
                import pandas as pd
                
                metadata_path = DATA_CONFIG['metadata_path']
                if not os.path.exists(metadata_path):
                    print(f"元数据文件不存在: {metadata_path}")
                    continue
                
                print(f"正在加载元数据: {metadata_path}")
                
                # 使用多编码回退机制读取CSV
                data = None
                for encoding in ['utf-8', 'gbk', 'latin-1']:
                    try:
                        data = pd.read_csv(metadata_path, encoding=encoding)
                        print(f"成功使用 {encoding} 编码读取元数据")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    print("无法读取元数据文件，请检查文件编码")
                    continue
                
                print(f"元数据加载成功，共 {len(data)} 条记录")
                print(f"列名: {list(data.columns)}")
                print(f"数据预览:\n{data.head()}")
                
                # 基本统计信息
                print(f"\n基本统计信息:")
                print(f"数据形状: {data.shape}")
                print(f"数据类型:\n{data.dtypes}")
                
                if 'avg_cell_value' in data.columns:
                    print(f"平均单元格值统计:\n{data['avg_cell_value'].describe()}")
                
                print("\n数据预处理完成！")
                
            except Exception as e:
                print(f"数据预处理过程中出现错误: {e}")
        elif choice == "3":
            print("\n开始数据集分割...")
            try:
                from config import DATA_CONFIG
                import pandas as pd
                import numpy as np
                from sklearn.model_selection import train_test_split
                
                # 检查元数据文件是否存在
                metadata_path = DATA_CONFIG['metadata_path']
                if not os.path.exists(metadata_path):
                    print(f"元数据文件不存在: {metadata_path}")
                    continue
                
                print(f"正在加载元数据: {metadata_path}")
                
                # 使用多编码回退机制读取CSV
                data = None
                for encoding in ['utf-8', 'gbk', 'latin-1']:
                    try:
                        data = pd.read_csv(metadata_path, encoding=encoding)
                        print(f"成功使用 {encoding} 编码读取元数据")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if data is None:
                    print("无法读取元数据文件，请检查文件编码")
                    continue
                
                # 划分数据集
                print("正在划分数据集...")
                train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
                val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
                
                print(f"数据集划分完成:")
                print(f"训练集: {len(train_data)} 条记录 ({len(train_data)/len(data)*100:.1f}%)")
                print(f"验证集: {len(val_data)} 条记录 ({len(val_data)/len(data)*100:.1f}%)")
                print(f"测试集: {len(test_data)} 条记录 ({len(test_data)/len(data)*100:.1f}%)")
                
                # 保存划分结果
                train_path = os.path.join('data', 'train_data.csv')
                val_path = os.path.join('data', 'val_data.csv')
                test_path = os.path.join('data', 'test_data.csv')
                
                train_data.to_csv(train_path, index=False, encoding='utf-8')
                val_data.to_csv(val_path, index=False, encoding='utf-8')
                test_data.to_csv(test_path, index=False, encoding='utf-8')
                
                print(f"划分结果已保存到:")
                print(f"  - 训练集: {train_path}")
                print(f"  - 验证集: {val_path}")
                print(f"  - 测试集: {test_path}")
                
                print("\n数据集划分完成！")
                
            except Exception as e:
                print(f"数据集分割过程中出现错误: {e}")
        elif choice == "4":
            print("\n开始模型测试...")
            try:
                # 简单的模型测试功能
                import torch
                from model import EnhancedSimAMResUNet
                from config import MODEL_CONFIG, DATA_CONFIG
                
                print("正在测试EnhancedSimAMResUNet模型...")
                
                # 创建模型实例
                model = EnhancedSimAMResUNet(
                    n_channels=DATA_CONFIG['sequence_length'],
                    n_classes=DATA_CONFIG['prediction_length'],
                    features=MODEL_CONFIG['features'],
                    bilinear=MODEL_CONFIG['bilinear'],
                    dropout=MODEL_CONFIG['dropout']
                )
                
                # 创建测试输入
                test_input = torch.randn(1, DATA_CONFIG['sequence_length'], 128, 128)
                
                # 前向传播测试
                with torch.no_grad():
                    output = model(test_input)
                
                print(f"模型测试成功！")
                print(f"输入形状: {test_input.shape}")
                print(f"输出形状: {output.shape}")
                print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
                
            except Exception as e:
                print(f"模型测试过程中出现错误: {e}")
        elif choice == "5":
            print("退出系统")
            sys.exit(0)
        else:
            print("无效选择，请输入1-5之间的数字")

if __name__ == "__main__":
    main()
