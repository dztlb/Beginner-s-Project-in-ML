# MgCo2O4电极材料机器学习项目

## 项目简介

本项目使用机器学习技术对MgCo2O4电极材料的性能进行预测和分析，包括比电容预测、特征重要性分析和形貌分类预测等任务。

## 项目结构

```
mgco2o4_project/
├── data/                              # 数据集目录
│   ├── data_aftercalculate.xlsx      # 主数据集
│   └── 数据集来源文献.xlsx           # 文献来源数据
├── config.py                          # 配置文件
├── 项目详细说明文档.md                 # 详细项目说明文档
├── requirements.txt                    # 项目依赖
├── main.py                            # 主程序
├── README.md                          # 项目说明
├── models/                             # 模型保存目录
├── results/                            # 结果保存目录
├── plots/                              # 图表保存目录
└── src/                               # 源代码目录
    ├── data_preprocessing.py          # 数据预处理模块
    ├── feature_selection.py           # 特征选择模块
    ├── model_training.py              # 模型训练模块
    └── shap_analysis.py               # SHAP分析模块
```

## 快速开始

### 1. 环境配置

#### 使用Anaconda（推荐）

```bash
# 创建conda环境
conda create -n mgco2o4_ml python=3.9
conda activate mgco2o4_ml

# 安装依赖
pip install -r requirements.txt
```

#### 使用PyCharm

1. 下载安装PyCharm
2. 创建新项目，选择之前创建的conda环境
3. 安装项目依赖包

### 2. 运行项目

#### 运行完整流程

```bash
python main.py
```

#### 交互式运行（选择模块）

```bash
python main.py --interactive
```

#### 运行单独模块

```bash
# 数据预处理
python src/data_preprocessing.py

# 特征选择
python src/feature_selection.py

# 模型训练
python src/model_training.py

# SHAP分析
python src/shap_analysis.py
```

## 主要功能

### 1. 数据预处理
- 数据加载和清洗
- 缺失值处理
- 分类特征编码
- 特征标准化

### 2. 特征选择
- 递归特征消除（RFE）
- 多种基模型支持（RF、XGBoost、DT）
- 特征重要性分析
- 性能评估和可视化

### 3. 模型训练
- 多种机器学习算法
- 交叉验证
- 超参数调优
- Stacking模型融合
- 性能评估和对比

### 4. SHAP分析
- 模型解释性分析
- 特征重要性可视化
- 依赖图分析
- 瀑布图分析

## 输出结果

项目运行完成后会生成以下文件：

- `feature_selection_results.xlsx` - 特征选择结果
- `training_results.xlsx` - 模型训练结果
- `shap_analysis_results.xlsx` - SHAP分析结果
- `models/` 目录 - 训练好的模型文件
- 各种可视化图表

## 技术特点

- **模块化设计**：每个功能模块独立，便于维护和扩展
- **多种算法支持**：支持随机森林、XGBoost、决策树等多种算法
- **完整的ML流程**：从数据预处理到模型解释的完整流程
- **可视化丰富**：提供多种图表和可视化结果
- **易于使用**：零基础用户也能快速上手

## 系统要求

### 最低配置
- CPU: Intel i5或AMD Ryzen 5（4核心以上）
- 内存: 8GB RAM
- 存储: 20GB可用空间

### 推荐配置
- CPU: Intel i7或AMD Ryzen 7（8核心以上）
- 内存: 16GB RAM
- 存储: 50GB可用空间（SSD推荐）

## 常见问题

### 1. 环境配置问题
- 确保使用Python 3.8-3.9版本
- 检查所有依赖包是否正确安装
- 确保conda环境已激活

### 2. 数据加载问题
- 检查Excel文件路径是否正确
- 确保文件格式为.xlsx
- 检查文件是否损坏

### 3. 内存不足问题
- 减少数据样本数量
- 关闭其他占用内存的程序
- 使用数据生成器处理大数据集

## 技术支持

如遇到问题，请检查：

1. 环境配置是否正确
2. 依赖包版本是否兼容
3. 数据文件格式是否正确
4. 系统资源是否充足

## 项目扩展

本项目可以进一步扩展：

1. 添加更多机器学习算法
2. 支持更多数据格式
3. 增加深度学习模型
4. 构建Web界面
5. 添加模型部署功能

## 许可证

本项目仅供学习和研究使用。

## 更新日志

- v1.0.0: 初始版本，包含完整的机器学习流程
- 支持数据预处理、特征选择、模型训练和SHAP分析
- 提供详细的文档和示例代码
