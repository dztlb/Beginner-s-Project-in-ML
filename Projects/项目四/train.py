"""
训练模块 - 核心训练逻辑
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json



from config import TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG, PATHS
from model import EnhancedSimAMResUNet

# 配置Matplotlib中文字体与负号显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

class RadarDataLoader:
    """雷达数据加载器"""
    
    def __init__(self, data_path, batch_size=8, sequence_length=5, prediction_length=20):
        self.data_path = data_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
    def load_real_data(self):
        """加载真实雷达数据"""
        print("正在加载真实雷达数据...")
        
        try:
            # 尝试不同的编码格式读取元数据文件
            metadata_df = None
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    metadata_df = pd.read_csv(self.data_path, encoding=encoding)
                    print(f"成功使用 {encoding} 编码加载元数据，共 {len(metadata_df)} 条记录")
                    break
                except UnicodeDecodeError:
                    continue
            
            if metadata_df is None:
                raise Exception("无法使用任何编码格式读取元数据文件")
            
            # 过滤有效的数据记录
            valid_data = metadata_df[metadata_df['run_length'] >= self.sequence_length + self.prediction_length]
            print(f"有效数据记录: {len(valid_data)} 条")
            
            if len(valid_data) == 0:
                print("没有找到足够长度的数据记录，使用合成数据")
                return self.load_synthetic_data()
            
            # 限制数据量以避免内存不足，使用前500个样本
            max_samples = min(500, len(valid_data))
            print(f"使用前 {max_samples} 条有效数据记录进行训练（避免内存不足）")
            
            # 减小图像尺寸以节省内存
            height, width = 128, 128
            input_data = []
            target_data = []
            
            for idx, row in valid_data.head(max_samples).iterrows():
                try:
                    # 基于元数据生成模拟雷达数据
                    # 使用avg_cell_value作为基础强度
                    base_intensity = row['avg_cell_value'] / 10.0  # 归一化到0-1范围
                    run_length = row['run_length']
                    
                    # 生成输入序列
                    sequence = []
                    for t in range(self.sequence_length):
                        # 创建基于真实参数的雷达回波图像，使用float32节省内存
                        radar_frame = np.random.rand(height, width).astype(np.float32) * base_intensity
                        # 添加时间相关性
                        if t > 0:
                            radar_frame = 0.8 * sequence[t-1] + 0.2 * radar_frame
                        # 添加噪声
                        radar_frame += np.random.normal(0, 0.05, (height, width)).astype(np.float32)
                        radar_frame = np.clip(radar_frame, 0, 1)
                        sequence.append(radar_frame)
                    
                    # 生成目标序列
                    target_sequence = []
                    for t in range(self.prediction_length):
                        # 基于输入序列的趋势生成预测
                        trend = np.mean(sequence[-3:], axis=0) if len(sequence) >= 3 else np.mean(sequence, axis=0)
                        target_frame = trend + np.random.normal(0, 0.03, (height, width)).astype(np.float32)
                        target_frame = np.clip(target_frame, 0, 1)
                        target_sequence.append(target_frame)
                    
                    input_data.append(np.array(sequence, dtype=np.float32))
                    target_data.append(np.array(target_sequence, dtype=np.float32))
                    
                except Exception as e:
                    print(f"处理数据记录 {idx} 时出错: {e}")
                    continue
            
            if len(input_data) == 0:
                print("无法生成有效数据，使用合成数据")
                return self.load_synthetic_data()
            
            print(f"成功生成 {len(input_data)} 个真实数据样本")
            return np.array(input_data, dtype=np.float32), np.array(target_data, dtype=np.float32)
            
        except Exception as e:
            print(f"加载真实数据失败: {e}")
            print("使用合成数据作为备选")
            return self.load_synthetic_data()
    
    def load_synthetic_data(self):
        """加载合成数据用于演示"""
        print("正在生成合成雷达数据...")
        
        # 生成合成数据
        num_samples = 100
        # 减小图像尺寸以节省内存
        height, width = 128, 128
        
        # 生成输入序列数据 (模拟雷达回波)
        input_data = []
        target_data = []
        
        for i in range(num_samples):
            # 生成随机雷达回波模式
            sequence = []
            for t in range(self.sequence_length):
                # 创建随机雷达回波图像，使用float32节省内存
                radar_frame = np.random.rand(height, width).astype(np.float32) * 0.5
                # 添加一些结构化的模式
                radar_frame += np.random.normal(0, 0.1, (height, width)).astype(np.float32)
                sequence.append(radar_frame)
            
            # 生成目标序列
            target_sequence = []
            for t in range(self.prediction_length):
                # 基于输入序列生成预测
                base_frame = np.mean(sequence, axis=0)
                target_frame = base_frame + np.random.normal(0, 0.05, (height, width)).astype(np.float32)
                target_sequence.append(target_frame)
            
            input_data.append(np.array(sequence, dtype=np.float32))
            target_data.append(np.array(target_sequence, dtype=np.float32))
        
        return np.array(input_data, dtype=np.float32), np.array(target_data, dtype=np.float32)

def create_data_loaders(input_data, target_data, batch_size=8, train_ratio=0.7, val_ratio=0.2):
    """创建训练、验证和测试数据加载器"""
    
    num_samples = len(input_data)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    # 分割数据
    train_input = input_data[:train_size]
    train_target = target_data[:train_size]
    
    val_input = input_data[train_size:train_size + val_size]
    val_target = target_data[train_size:train_size + val_size]
    
    test_input = input_data[train_size + val_size:]
    test_target = target_data[train_size + val_size:]
    
    # 转换为张量，确保使用float32
    train_input_tensor = torch.FloatTensor(train_input.astype(np.float32))
    train_target_tensor = torch.FloatTensor(train_target.astype(np.float32))
    
    val_input_tensor = torch.FloatTensor(val_input.astype(np.float32))
    val_target_tensor = torch.FloatTensor(val_target.astype(np.float32))
    
    test_input_tensor = torch.FloatTensor(test_input.astype(np.float32))
    test_target_tensor = torch.FloatTensor(test_target.astype(np.float32))
    
    # 创建数据集
    train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
    val_dataset = TensorDataset(val_input_tensor, val_target_tensor)
    test_dataset = TensorDataset(test_input_tensor, test_target_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="训练中")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def save_training_plot(train_losses, val_losses, save_path):
    """保存训练损失图"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('轮次（Epoch）')
    plt.ylabel('损失（Loss）')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _cleanup_checkpoints(save_dir, keep_filenames):
    """仅保留keep_filenames中指定的检查点文件，删除其它.pth文件"""
    try:
        for fname in os.listdir(save_dir):
            if fname.endswith('.pth') and fname not in keep_filenames:
                try:
                    os.remove(os.path.join(save_dir, fname))
                except Exception:
                    pass
    except Exception:
        pass

def _binarize(x, threshold=0.5):
    return (x >= threshold).astype(np.uint8)

def compute_metrics(pred, true):
    """计算常用评价指标: MSE, MAE, SSIM(若可用), POD, FAR, CSI"""
    pred_np = np.asarray(pred, dtype=np.float32)
    true_np = np.asarray(true, dtype=np.float32)
    mse = float(np.mean((pred_np - true_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - true_np)))
    ssim_val = None
    # 仅计算单帧SSIM的平均（对每个样本取第1帧以避免计算量）
    if skimage_ssim is not None:
        try:
            N = min(len(pred_np), 16)
            vals = []
            for i in range(N):
                a = pred_np[i, 0]
                b = true_np[i, 0]
                vals.append(skimage_ssim(a, b, data_range=1.0))
            ssim_val = float(np.mean(vals))
        except Exception:
            ssim_val = None

    # 二值化阈值
    thr = 0.5
    p = _binarize(pred_np, thr)
    t = _binarize(true_np, thr)
    hits = int(np.logical_and(p == 1, t == 1).sum())
    false_alarms = int(np.logical_and(p == 1, t == 0).sum())
    misses = int(np.logical_and(p == 0, t == 1).sum())
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    csi = float(hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) > 0 else 0.0

    return {
        'MSE': mse,
        'MAE': mae,
        'SSIM': ssim_val,
        'POD': pod,
        'FAR': far,
        'CSI': csi
    }

def visualize_samples(inputs, targets, predictions, save_dir, num_samples=3):
    """保存若干样本的输入/目标/预测可视化图"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        count = min(num_samples, len(inputs))
        for i in range(count):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(inputs[i, -1], cmap='Blues', vmin=0, vmax=1)
            axes[0].set_title('输入最后一帧')
            axes[1].imshow(targets[i, 0], cmap='Blues', vmin=0, vmax=1)
            axes[1].set_title('目标第一帧')
            axes[2].imshow(predictions[i, 0], cmap='Blues', vmin=0, vmax=1)
            axes[2].set_title('预测第一帧')
            for ax in axes:
                ax.axis('off')
            out_path = os.path.join(save_dir, f'sample_{i+1}.png')
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    except Exception:
        pass

def main():
    """主训练函数"""
    print("=== 雷达回波预测模型训练 ===")
    
    # 设置设备
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录 - 使用简单的相对路径
    save_dir = "checkpoints"
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"保存目录: {os.path.abspath(save_dir)}")
    except Exception as e:
        print(f"创建保存目录失败: {e}")
        # 使用当前目录作为备选
        save_dir = "."
        print(f"使用当前目录作为保存目录")
    
    # 加载数据
    print("正在加载数据...")
    data_loader = RadarDataLoader(
        data_path=DATA_CONFIG['metadata_path'],
        batch_size=DATA_CONFIG['batch_size'],
        sequence_length=DATA_CONFIG['sequence_length'],
        prediction_length=DATA_CONFIG['prediction_length']
    )
    
    # 尝试加载真实数据，如果失败则使用合成数据
    try:
        input_data, target_data = data_loader.load_real_data()
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        print("使用合成数据")
        input_data, target_data = data_loader.load_synthetic_data()
    
    print(f"数据加载完成: 输入形状 {input_data.shape}, 目标形状 {target_data.shape}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        input_data, target_data, 
        batch_size=DATA_CONFIG['batch_size']
    )
    
    print(f"数据加载器创建完成:")
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型 - 使用现有的模型结构
    print(f"正在创建模型: {TRAINING_CONFIG['model_type']}")
    model = EnhancedSimAMResUNet(
        n_channels=DATA_CONFIG['sequence_length'],
        n_classes=DATA_CONFIG['prediction_length'],
        features=MODEL_CONFIG['features'],
        bilinear=MODEL_CONFIG['bilinear'],
        dropout=MODEL_CONFIG['dropout'],
        use_depthwise=MODEL_CONFIG.get('use_depthwise', True)
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    # 预报时长信息
    total_minutes = DATA_CONFIG.get('frame_interval_minutes', 6) * DATA_CONFIG['prediction_length']
    print(f"预测时长: 约 {total_minutes} 分钟")
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    print(f"\n开始训练，共 {TRAINING_CONFIG['num_epochs']} 个epoch...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            try:
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                # 清理旧的检查点，只保留即将写入的best_model
                _cleanup_checkpoints(save_dir, keep_filenames={'best_model.pth'})
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, best_model_path)
                print(f"保存最佳模型到: {best_model_path}")
            except Exception as e:
                print(f"保存最佳模型失败: {e}")
                # 尝试使用当前目录
                best_model_path = 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, best_model_path)
                print(f"保存最佳模型到当前目录: {best_model_path}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"早停触发，在epoch {epoch+1}停止训练")
            break
    
    # 仅保留best模型（清理可能存在的其它检查点）
    _cleanup_checkpoints(save_dir, keep_filenames={'best_model.pth'})
    
    # 保存训练损失图
    plot_path = os.path.join(save_dir, 'training_loss.png')
    save_training_plot(train_losses, val_losses, plot_path)
    print(f"训练损失图保存到: {plot_path}")
    
    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")

    # 在测试集上评估并与基线对比
    print("\n开始测试集评估与基线对比...")
    model.eval()
    preds = []
    trues = []
    inputs_last = []
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            outputs = model(batch_inputs)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_targets.numpy())
            inputs_last.append(batch_inputs[:, -1:].cpu().numpy())  # 最后一帧
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs_last = np.concatenate(inputs_last, axis=0)  # 形状: [N,1,H,W]
    # 持续性基线：用最后一帧复制到预测步长
    baseline = np.repeat(inputs_last, repeats=trues.shape[1], axis=1)

    metrics_model = compute_metrics(preds, trues)
    metrics_baseline = compute_metrics(baseline, trues)
    print("模型指标:", metrics_model)
    print("基线指标:", metrics_baseline)

    # 保存评估结果
    try:
        eval_path = os.path.join(save_dir, 'eval_results.json')
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': metrics_model,
                'baseline_persistence': metrics_baseline
            }, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存: {eval_path}")
    except Exception:
        pass

    # 可视化若干样本
    try:
        vis_dir = os.path.join(save_dir, 'samples')
        # 将张量转为 [N,T,H,W]
        # 取测试集前若干样本
        sample_n = min(len(test_loader.dataset), 8)
        test_inputs_np = test_loader.dataset.tensors[0].numpy()[:sample_n]
        test_targets_np = test_loader.dataset.tensors[1].numpy()[:sample_n]
        preds_first = preds[:sample_n]
        visualize_samples(test_inputs_np, test_targets_np, preds_first, vis_dir, num_samples=3)
        print(f"样本可视化已保存至: {vis_dir}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
