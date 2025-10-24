# 风电功率预测项目使用说明

## 一、运行方式
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   或使用 Conda：
   ```bash
   conda env create -f environment.yml
   conda activate wind_power_prediction
   ```

2. 确保 Python 版本 ≥ 3.8，并且已安装 TensorFlow（requirements 已包含）。

---

## 二、放置数据
将数据文件命名为 **`data.csv`**，放在项目根目录（与 `run.py` 同一层）。  
文件应包含以下字段：

| 字段 | 含义 |
|------|------|
| DATETIME | 时间戳 |
| WINDDIRECTION | 风向 |
| TEMPERATURE | 温度 |
| HUMIDITY | 湿度 |
| PRESSURE | 气压 |
| WINDSPEED | 风速 |
| POWER | 实际输出功率（预测目标） |

---

## 三、运行主程序
在项目根目录执行：
```bash
python run.py --csv data.csv --datetime DATETIME --target POWER --save_prefix run1
```

---

## 四、可选参数

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `--test_size` | 0.2 | 测试集占比 |
| `--time_steps` | 10 | 深度学习模型时间窗口长度 |
| `--save_prefix` | run1 | 输出结果前缀名称 |

示例：
```bash
python run.py --csv data.csv --test_size 0.25 --time_steps 12 --save_prefix testA
```

---

## 五、输出结果说明

程序运行结束后，会在 `results/` 目录生成以下内容：

| 路径 | 内容说明 |
|------|-----------|
| `results/logs/run1_kpss.json` | KPSS 平稳性检验结果 |
| `results/logs/run1_selected_features.json` | RFECV 选择的特征 |
| `results/logs/run1_rf.json` | 随机森林模型最优参数与评估指标 |
| `results/logs/run1_summary.json` | 随机森林与深度学习模型性能对比 |
| `results/models/run1_dl.keras` | 深度学习模型权重文件 |
| `results/plots/` | 图表输出（可选） |

**典型输出指标示例：**
```json
{
  "rf": {"MSE": 0.1345, "RMSE": 0.3667, "MAE": 0.2456, "R2": 0.8654},
  "dl": {"MSE": 0.0987, "RMSE": 0.3142, "MAE": 0.1987, "R2": 0.9123}
}
```

---

运行完成后，模型及日志均会保存在 `results/` 文件夹中，可直接查看。
