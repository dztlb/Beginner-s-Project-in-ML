#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行训练并查看结果的脚本
"""

import subprocess
import sys
import os

def run_training():
    """运行训练程序"""
    print("开始运行风电功率预测训练...")
    print("=" * 60)
    
    try:
        # 运行主程序，选择快速验证模式
        process = subprocess.Popen(
            [sys.executable, 'wind_power_prediction.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 发送选择1（快速验证模式）
        stdout, stderr = process.communicate(input='1\n')
        
        print("程序输出:")
        print(stdout)
        
        if stderr:
            print("错误信息:")
            print(stderr)
        
        # 检查生成的图表文件
        print("\n" + "=" * 60)
        print("检查生成的图表文件:")
        
        chart_files = [
            '相关性热力图.png',
            '深度学习训练历史.png',
            '随机森林优化收敛曲线.png',
            '模型对比结果.png'
        ]
        
        for file in chart_files:
            if os.path.exists(file):
                print(f"✓ {file} - 已生成")
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"  文件大小: {file_size:.1f} KB")
            else:
                print(f"✗ {file} - 未生成")
        
        print("\n训练完成！")
        print("如果图表文件已生成，你可以在项目目录中找到它们。")
        
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    run_training()
