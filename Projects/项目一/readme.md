下载项目文件后，在pycharm中将 项目一 文件夹作为项目打开

按照 requirements.txt 配置环境

运行 main.py

具体步骤：

```
在Pycharm 或 VScode中打开终端 或 打开 Anaconda Prompt（anaconda提供的终端）

为该项目创建新虚拟环境 
conda create --name ml_env python=3.10

进入  ml_env 环境
conda activate ml_env

配置环境（进入项目一文件夹执行）
比如：(ml_env) PS C:\Users\31362\Desktop\important file\BPML\Beginner-s-Project-in-ML\Projects\项目一>
如果你已经在在Pycharm 或 VScode中打开项目一，则打开终端自动进入，否则自己在终端进入
pip install -r requirements.txt

运行项目
python main.py
```
