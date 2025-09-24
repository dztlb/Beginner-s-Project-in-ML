.PHONY: help install run clean test lint format

help: ## 显示帮助信息
	@echo "MgCo2O4电极材料机器学习项目"
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装项目依赖
	pip install -r requirements.txt

install-dev: ## 安装开发依赖
	pip install -e .[dev]

run: ## 运行主程序
	python main.py

run-interactive: ## 交互式运行
	python main.py --interactive

test: ## 运行测试
	pytest tests/ -v

lint: ## 代码检查
	flake8 src/ main.py
	mypy src/ main.py

format: ## 代码格式化
	black src/ main.py
	isort src/ main.py

clean: ## 清理临时文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete

clean-results: ## 清理结果文件
	rm -rf results/*
	rm -rf models/*
	rm -rf plots/*

setup: ## 项目初始化
	mkdir -p data models plots results docs scripts tests
	touch data/.gitkeep models/.gitkeep plots/.gitkeep results/.gitkeep docs/.gitkeep scripts/.gitkeep tests/.gitkeep

check-data: ## 检查数据文件
	python scripts/check_data_types.py
	python scripts/check_target_distribution.py
