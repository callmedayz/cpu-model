"""
Setup script for Intel-Optimized AI Model
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="intel-optimized-ai-model",
    version="1.0.0",
    author="Intel AI Model Project",
    author_email="veiledmirage@gmail.com",
    description="Intel-optimized AI model for CPU-only deployment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/callmedayz/cpu-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "intel-ai-chat=src.intel_chat_system:main",
            "intel-ai-train=src.intel_training_loop:main",
            "intel-ai-eval=src.intel_model_evaluator:main",
            "intel-ai=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/callmedayz/cpu-model/issues",
        "Source": "https://github.com/callmedayz/cpu-model",
        "Documentation": "https://github.com/callmedayz/cpu-model/blob/main/README.md",
    },
)
