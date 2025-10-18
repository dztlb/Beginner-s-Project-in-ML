[anaconda下载](https://www.anaconda.com/download)

[anaconda安装教程](https://www.cnblogs.com/singleYao/p/13475709.html)

[anaconda视频教程](https://www.bilibili.com/video/BV1ywpgz3EZv/?spm_id_from=333.337.search-card.all.click&vd_source=7bd20288510615ec72aca912fe510a70)

安装完成后请使用 anaconda prompt 配置环境而非图形界面。

注意：不要更改bash环境，请创建新的环境并在新环境进行项目依赖的配置。你的anaconda本身运行在bash环境，更改bash会造成不可预知的问题。

国内从官方源下载机器学习库很慢，在根据项目依赖配置环境前请换源。[换源教程](https://www.cnblogs.com/zhouchengzhi/p/18163694)

anaconda常用命令：

```python
# 创建一个名为test的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）

conda create --name test python=3.4

# 安装好后，使用activate激活某个环境

conda activate test

# 查看python版本

python --version

# 退出当前虚拟环境 返回base环境

conda deactivate

# 删除一个已有的环境

conda remove -n 环境名 --all

# 查看当前机器中的虚拟环境

conda env list
```

在终端中还会涉及到常见的文件操作

| 功能 | macOS / Linux 命令 | Windows 命令（CMD / PowerShell） | 说明 |
|------|----------------------|----------------------------------|------|
| 查看当前路径 | `pwd` | `cd` | 打印当前工作目录 |
| 查看文件列表 | `ls` 或 `ls -l` | `dir` | 查看当前文件夹内容 |
| 进入子文件夹 | `cd 文件夹名` | `cd 文件夹名` | 切换到指定文件夹 |
| 返回上一级文件夹 | `cd ..` | `cd ..` | 返回上层目录 |
| 返回用户主目录 | `cd ~` | `cd %HOMEPATH%` | 快速回主目录 |
| 创建文件夹 | `mkdir 文件夹名` | `mkdir 文件夹名` | 创建新目录 |
| 删除文件夹 | `rm -r 文件夹名` | `rmdir /s 文件夹名` | 删除整个文件夹（⚠️小心使用） |
| 创建空文件 | `touch 文件名` | `type nul > 文件名` 或 `echo. > 文件名` | 创建空文件 |
| 查看文件内容 | `cat 文件名` | `type 文件名` | 显示文件内容 |
| 分页查看文件内容 | `less 文件名` 或 `more 文件名` | `more 文件名` | 适合长文件阅读 |
| 复制文件 | `cp 源 目标` | `copy 源 目标` | 复制文件 |
| 移动或重命名文件 | `mv 源 目标` | `move 源 目标` | 移动或改名 |
| 删除文件 | `rm 文件名` | `del 文件名` | 删除文件 |