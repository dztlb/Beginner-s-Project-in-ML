### vscode配置python环境的步骤与pycharm类似

vscode下载python扩展，将python解释器配置为anaconda环境即可

### 这里我们重点关注 vscode 的 git 配置

[给傻子的Git教程](https://www.bilibili.com/video/BV1Hkr7YYEh8?spm_id_from=333.788.videopod.sections&vd_source=7bd20288510615ec72aca912fe510a70)

# 🚀 VS Code 配置 GitHub 的完整流程

## 🔹 一、前置准备（只做一次）

1. **安装 Git**
    
    - [下载 Git](https://git-scm.com/downloads) 并安装。
        
    - 验证是否成功：
        
        ```bash
        git --version
        ```
        
2. **配置 Git 用户信息**
    
    ```bash
    git config --global user.name "你的GitHub用户名"
    git config --global user.email "你的GitHub绑定邮箱"
    ```
    
3. **VS Code 安装插件**
    
    - 打开扩展（Ctrl+Shift+X）
        
    - 安装：
        
        - ✅ GitHub Pull Requests and Issues
            
        - ✅ GitLens（可选）
            
4. **在 VS Code 登录 GitHub**
    
    - 打开命令面板（Ctrl+Shift+P） → 输入 `GitHub: Sign in`
        
    - 按提示跳转浏览器授权登录
        

---

## 🔹 二、流程 A：本地项目推送到 GitHub

1. **在 VS Code 打开本地项目**
    
    ```bash
    cd 你的项目目录
    code .
    ```
    
2. **初始化 Git 仓库**
    
    - 左侧 **Source Control** → `Initialize Repository`  
        （等价于 `git init`）
        
3. **第一次提交**
    
    - 修改/保存文件后 → 在 Source Control 面板输入提交信息 → 点 ✔（Commit）
        
4. **推送到 GitHub**  
    有两种方式：
    
    - **自动发布**：点击 VS Code 提示的 `Publish to GitHub` → 选择 Public/Private → 自动建仓库并推送
        
    - **手动推送**：
        
        1. 先在 GitHub 网站上建一个空仓库
            
        2. 在终端运行：
            
            ```bash
            git remote add origin https://github.com/你的用户名/仓库名.git
            git branch -M main
            git push -u origin main
            ```
            
5. **后续更新**
    
    - 每次修改后：
        
        ```bash
        git add .
        git commit -m "更新说明"
        git push
        ```
        
    
    或直接在 Source Control 面板点击 “Commit & Sync”。
    

---

## 🔹 三、流程 B：从 GitHub 克隆项目到本地并编辑

1. **复制仓库地址**
    
    - 在 GitHub 仓库页面，点击绿色按钮 `Code` → 复制 HTTPS 或 SSH 地址
        
    - 例如：
        
        ```
        https://github.com/用户名/仓库名.git
        ```
        
2. **克隆项目到本地**  
    在终端执行：
    
    ```bash
    git clone https://github.com/用户名/仓库名.git
    cd 仓库名
    code .
    ```
    
3. **编辑并保存代码**
    
    - 在 VS Code 修改文件
        
    - 使用 Source Control 或命令行提交
        
4. **提交 & 推送修改**
    
    ```bash
    git add .
    git commit -m "修改说明"
    git push
    ```
    

---

## 🔹 四、总结对比

|场景|关键步骤|
|---|---|
|**本地 → GitHub**|`git init` → commit → 连接远程仓库 → `git push`|
|**GitHub → 本地**|`git clone` → 编辑 → commit → `git push`|
