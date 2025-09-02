[anaconda下载](https://www.anaconda.com/download)

[anaconda安装教程](https://www.cnblogs.com/singleYao/p/13475709.html)

安装完成后请使用 anaconda prompt 配置环境而非图形界面。

注意：不要更改bash环境，请创建新的环境并在新环境进行项目依赖的配置。你的anaconda本身运行在bash环境，更改bash会造成不可预知的问题。

国内从官方源下载机器学习库很慢，在根据项目依赖配置环境前请换源。[换源教程](https://www.cnblogs.com/zhouchengzhi/p/18163694)

anaconda常用命令：

```python
# 创建一个名为python34的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）

conda create --name python34 python=3.4

# 安装好后，使用activate激活某个环境

activate python34 # for Windows

source activate python34 # for Linux & Mac

# 激活后，会发现terminal输入的地方多了python34的字样，实际上，此时系统做的事情就是把默认2.7环境从PATH中去除，再把3.4对应的命令加入PATH

# 此时，再次输入

python --version

# 可以得到`Python 3.4.5 :: Anaconda 4.1.1 (64-bit)`，即系统已经切换到了3.4的环境

# 如果想返回默认的python 2.7环境，运行

deactivate python34 # for Windows

source deactivate python34 # for Linux & Mac

# 删除一个已有的环境

conda remove --name python34 --all
```
