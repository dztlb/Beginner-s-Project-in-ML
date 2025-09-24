"""
字体配置文件
设置matplotlib中文字体支持，解决中文显示为框框的问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

def setup_chinese_fonts():
    """
    设置中文字体支持
    
    Returns:
        bool: 是否成功设置中文字体
    """
    try:
        # 尝试设置中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'STSong', 'STKaiti']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 找到可用的中文字体
        selected_font = None
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"中文字体设置成功，使用字体: {selected_font}")
            return True
        else:
            # 如果没有找到中文字体，尝试使用系统默认字体
            print("未找到中文字体，尝试使用系统默认字体...")
            plt.rcParams['axes.unicode_minus'] = False
            return False
            
    except Exception as e:
        print(f"字体设置失败: {e}")
        return False

def get_available_fonts():
    """获取系统中可用的字体列表"""
    try:
        fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = [f for f in fonts if any(c in f for c in ['黑', '宋', '楷', '仿宋', '微软', 'Sim', 'ST'])]
        return fonts, chinese_fonts
    except Exception as e:
        print(f"获取字体列表失败: {e}")
        return [], []

# 自动设置字体
if __name__ == "__main__":
    setup_chinese_fonts()
    all_fonts, chinese_fonts = get_available_fonts()
    print(f"\n系统中可用字体总数: {len(all_fonts)}")
    print(f"中文字体数量: {len(chinese_fonts)}")
    if chinese_fonts:
        print("可用的中文字体:")
        for font in chinese_fonts[:10]:  # 只显示前10个
            print(f"  - {font}")
