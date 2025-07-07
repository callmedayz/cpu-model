# Intel-Optimized AI Model - Makefile
# Common development tasks

.PHONY: help install install-dev test chat train evaluate benchmark clean lint format setup

# Default target
help:
	@echo "Intel-Optimized AI Model - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make setup        - Complete project setup (venv + install)"
	@echo "  make install      - Install package and dependencies"
	@echo "  make install-dev  - Install with development dependencies"
	@echo ""
	@echo "Usage:"
	@echo "  make chat         - Start interactive chat system"
	@echo "  make train        - Run training pipeline"
	@echo "  make evaluate     - Run model evaluation"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo "  make test         - Run all tests"
	@echo ""
	@echo "Development:"
	@echo "  make lint         - Run code linting"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean temporary files"
	@echo ""

# Setup virtual environment and install
setup:
	python -m venv ai_env
	ai_env\Scripts\activate && pip install --upgrade pip
	ai_env\Scripts\activate && pip install -r requirements.txt
	@echo "✅ Setup complete! Activate with: ai_env\Scripts\activate"

# Install package
install:
	pip install -r requirements.txt
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# Run interactive chat
chat:
	python main.py chat

# Run training
train:
	python main.py train

# Run evaluation
evaluate:
	python main.py evaluate

# Run benchmarks
benchmark:
	python main.py benchmark

# Run tests
test:
	python main.py test

# Code linting
lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

# Code formatting
format:
	black src/ tests/ main.py setup.py --line-length=100

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Show project structure
tree:
	@echo "Project Structure:"
	@tree /F /A || dir /S /B

# Quick start guide
quickstart:
	@echo "🚀 Intel-Optimized AI Model - Quick Start"
	@echo ""
	@echo "1. Setup environment:"
	@echo "   make setup"
	@echo ""
	@echo "2. Activate environment:"
	@echo "   ai_env\Scripts\activate"
	@echo ""
	@echo "3. Start chatting:"
	@echo "   make chat"
	@echo ""
	@echo "4. Or use main interface:"
	@echo "   python main.py --help"
