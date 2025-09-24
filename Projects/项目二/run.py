#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风电功率预测项目运行脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    main()
