"""
Intel-Optimized AI Model Package

This package contains Intel-optimized implementations for AI model training,
evaluation, and deployment on CPU-only hardware.

Modules:
    intel_optimized_model: Core GPT-2 model with Intel optimizations
    intel_memory_optimizer: Memory management and optimization utilities
    intel_data_pipeline: Efficient data loading and preprocessing
    intel_training_loop: Complete training implementation
    intel_model_evaluator: Comprehensive model evaluation system
    intel_chat_system: Production-ready chat interface
"""

__version__ = "1.0.0"
__author__ = "Intel AI Model Project"
__description__ = "Intel-optimized AI model for CPU-only deployment"

# Import main classes for easy access
from .intel_optimized_model import IntelOptimizedGPT2
from .intel_memory_optimizer import MemoryOptimizedGPT2, IntelMemoryOptimizer
from .intel_data_pipeline import IntelOptimizedTextDataset, IntelDataPipeline
from .intel_training_loop import IntelOptimizedTrainer
from .intel_model_evaluator import IntelModelEvaluator
from .intel_chat_system import IntelChatSystem

__all__ = [
    'IntelOptimizedGPT2',
    'MemoryOptimizedGPT2',
    'IntelMemoryOptimizer',
    'IntelOptimizedTextDataset',
    'IntelDataPipeline',
    'IntelOptimizedTrainer',
    'IntelModelEvaluator',
    'IntelChatSystem'
]
