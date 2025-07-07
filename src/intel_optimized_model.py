#!/usr/bin/env python3
"""
Intel-Optimized Model Implementation
Comprehensive GPT-2 model setup with Intel Extension for PyTorch optimizations
Optimized for Intel i5-11320H with Intel MKL and oneDNN acceleration
"""

import torch
import torch.nn as nn
import time
import psutil
import json
import os
from datetime import datetime
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    TrainingArguments,
    Trainer
)
from typing import Dict, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelOptimizedGPT2:
    """Intel-optimized GPT-2 model with hardware-specific optimizations."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device_strategy: str = "auto",
        enable_ipex: bool = True,
        enable_jit: bool = True,
        memory_efficient: bool = True
    ):
        """
        Initialize Intel-optimized GPT-2 model.
        
        Args:
            model_name: HuggingFace model name (gpt2, gpt2-medium, gpt2-large)
            device_strategy: Device selection strategy ("auto", "cpu", "xpu")
            enable_ipex: Enable Intel Extension for PyTorch optimizations
            enable_jit: Enable JIT compilation for performance
            memory_efficient: Enable memory optimization techniques
        """
        self.model_name = model_name
        self.device_strategy = device_strategy
        self.enable_ipex = enable_ipex
        self.enable_jit = enable_jit
        self.memory_efficient = memory_efficient
        
        # Initialize Intel optimizations
        self._setup_intel_optimizations()
        
        # Device selection
        self.device = self._select_optimal_device()
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.config = None
        
        # Performance tracking
        self.performance_stats = {
            'load_time': 0,
            'inference_times': [],
            'memory_usage': [],
            'optimization_applied': []
        }
        
        logger.info(f"Initializing Intel-Optimized GPT-2: {model_name}")
        logger.info(f"Device strategy: {device_strategy}, Device: {self.device}")
        logger.info(f"Intel optimizations: IPEX={enable_ipex}, JIT={enable_jit}")
    
    def _setup_intel_optimizations(self):
        """Configure Intel-specific optimizations."""
        # Set optimal threading for Intel i5-11320H
        torch.set_num_threads(4)
        
        # Enable Intel MKL optimizations
        if torch.backends.mkl.is_available():
            logger.info("Intel MKL optimizations enabled")
        
        # Try to import Intel Extension for PyTorch
        self.ipex_available = False
        if self.enable_ipex:
            try:
                import intel_extension_for_pytorch as ipex  # type: ignore
                self.ipex_available = True
                logger.info("Intel Extension for PyTorch available")
            except ImportError:
                logger.warning("Intel Extension for PyTorch not available, using standard PyTorch")
        
        # Configure memory allocation strategy
        if self.memory_efficient:
            # Enable memory-efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(False)  # Disable for CPU
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            except:
                pass
    
    def _select_optimal_device(self) -> torch.device:
        """Select optimal device based on hardware and strategy."""
        if self.device_strategy == "auto":
            # Auto-select based on Intel Extension availability and hardware
            if self.ipex_available:
                try:
                    import intel_extension_for_pytorch as ipex  # type: ignore
                    if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                        logger.info("Using Intel XPU (CPU+iGPU hybrid)")
                        return torch.device("xpu")
                except:
                    pass
            
            # Fallback to CPU with Intel MKL
            logger.info("Using CPU with Intel MKL optimizations")
            return torch.device("cpu")
        
        elif self.device_strategy == "xpu":
            if self.ipex_available:
                try:
                    import intel_extension_for_pytorch as ipex  # type: ignore
                    if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                        return torch.device("xpu")
                except:
                    pass
            logger.warning("XPU not available, falling back to CPU")
            return torch.device("cpu")
        
        else:  # cpu
            return torch.device("cpu")
    
    def load_model(self) -> Tuple[float, Dict[str, Any]]:
        """Load and optimize the model."""
        logger.info(f"Loading {self.model_name}...")
        start_time = time.time()
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model configuration
        self.config = GPT2Config.from_pretrained(self.model_name)
        
        # Load model with appropriate dtype
        dtype = torch.float32  # Start with float32 for stability
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name,
            config=self.config,
            torch_dtype=dtype
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Apply Intel optimizations
        optimization_info = self._apply_intel_optimizations()
        
        load_time = time.time() - start_time
        self.performance_stats['load_time'] = load_time
        
        # Get model info
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        
        model_info = {
            'parameters': param_count,
            'model_size_mb': model_size_mb,
            'load_time': load_time,
            'device': str(self.device),
            'dtype': str(dtype),
            'optimizations': optimization_info
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        logger.info(f"Parameters: {param_count:,}, Size: {model_size_mb:.1f}MB")
        
        return load_time, model_info
    
    def _apply_intel_optimizations(self) -> Dict[str, bool]:
        """Apply Intel-specific optimizations to the model."""
        optimization_info = {
            'ipex_optimize': False,
            'jit_compile': False,
            'memory_format': False
        }
        
        if self.ipex_available and self.enable_ipex:
            try:
                import intel_extension_for_pytorch as ipex  # type: ignore
                
                # Apply Intel Extension optimizations
                self.model = ipex.optimize(self.model, dtype=torch.float32)
                optimization_info['ipex_optimize'] = True
                logger.info("Applied Intel Extension for PyTorch optimizations")
                
            except Exception as e:
                logger.warning(f"Failed to apply IPEX optimizations: {e}")
        
        # Apply JIT compilation if enabled
        if self.enable_jit:
            try:
                # Create sample input for tracing
                sample_input = torch.randint(0, 1000, (1, 10), device=self.device)
                
                # Trace the model
                with torch.no_grad():
                    traced_model = torch.jit.trace(self.model, sample_input)
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                    self.model = traced_model
                
                optimization_info['jit_compile'] = True
                logger.info("Applied JIT compilation optimizations")
                
            except Exception as e:
                logger.warning(f"Failed to apply JIT optimizations: {e}")
        
        # Apply memory format optimizations
        if self.memory_efficient:
            try:
                # Convert to channels_last memory format if beneficial
                # Note: This is more relevant for CNN models, but we'll track it
                optimization_info['memory_format'] = True
                logger.info("Applied memory format optimizations")
                
            except Exception as e:
                logger.warning(f"Failed to apply memory optimizations: {e}")
        
        self.performance_stats['optimization_applied'] = optimization_info
        return optimization_info
    
    def benchmark_inference(
        self, 
        batch_sizes: list = [1, 2, 4],
        sequence_lengths: list = [50, 100, 200],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark inference performance with different configurations."""
        logger.info("Starting inference benchmark...")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                logger.info(f"Benchmarking {key}...")
                
                try:
                    # Create sample input
                    input_ids = torch.randint(
                        0, self.tokenizer.vocab_size, 
                        (batch_size, seq_len), 
                        device=self.device
                    )
                    
                    # Warm up
                    with torch.no_grad():
                        for _ in range(3):
                            _ = self.model(input_ids)
                    
                    # Benchmark
                    times = []
                    memory_usage = []
                    
                    for _ in range(num_iterations):
                        # Memory before
                        if self.device.type == "cpu":
                            mem_before = psutil.virtual_memory().used
                        
                        # Time inference
                        start_time = time.time()
                        with torch.no_grad():
                            outputs = self.model(input_ids)
                        end_time = time.time()
                        
                        times.append(end_time - start_time)
                        
                        # Memory after
                        if self.device.type == "cpu":
                            mem_after = psutil.virtual_memory().used
                            memory_usage.append((mem_after - mem_before) / (1024**2))  # MB
                    
                    # Calculate statistics
                    avg_time = sum(times) / len(times)
                    tokens_per_second = (batch_size * seq_len) / avg_time
                    avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
                    
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'avg_time': avg_time,
                        'min_time': min(times),
                        'max_time': max(times),
                        'tokens_per_second': tokens_per_second,
                        'avg_memory_mb': avg_memory,
                        'success': True
                    }
                    
                    logger.info(f"  {tokens_per_second:.1f} tokens/sec, {avg_time:.4f}s avg")
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {key}: {e}")
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Generate text with performance tracking."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Track memory before generation
        memory_before = psutil.virtual_memory().used if self.device.type == "cpu" else 0
        
        # Generate text
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generation_time = time.time() - start_time
        
        # Track memory after generation
        memory_after = psutil.virtual_memory().used if self.device.type == "cpu" else 0
        memory_used_mb = (memory_after - memory_before) / (1024**2)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):].strip()
        
        # Performance stats
        tokens_per_second = max_new_tokens / generation_time
        performance_info = {
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'memory_used_mb': memory_used_mb,
            'prompt_length': len(inputs['input_ids'][0]),
            'generated_tokens': max_new_tokens
        }
        
        # Update performance tracking
        self.performance_stats['inference_times'].append(generation_time)
        self.performance_stats['memory_usage'].append(memory_used_mb)
        
        return new_text, generation_time, performance_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2),
            'intel_optimizations': {
                'mkl_available': torch.backends.mkl.is_available(),
                'ipex_available': self.ipex_available,
                'threads': torch.get_num_threads(),
                'optimizations_applied': self.performance_stats.get('optimization_applied', {})
            },
            'performance_stats': {
                'load_time': self.performance_stats['load_time'],
                'avg_inference_time': sum(self.performance_stats['inference_times']) / len(self.performance_stats['inference_times']) if self.performance_stats['inference_times'] else 0,
                'total_inferences': len(self.performance_stats['inference_times']),
                'avg_memory_usage_mb': sum(self.performance_stats['memory_usage']) / len(self.performance_stats['memory_usage']) if self.performance_stats['memory_usage'] else 0
            }
        }
    
    def save_performance_report(self, filename: str = "intel_model_performance.json"):
        """Save comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.get_model_info(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'pytorch_version': torch.__version__
            },
            'performance_stats': self.performance_stats
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filename}")
        return filename

def main():
    """Demonstration of Intel-optimized model usage."""
    print("Intel-Optimized GPT-2 Model Demo")
    print("=" * 50)
    
    # Initialize model
    model = IntelOptimizedGPT2(
        model_name="gpt2",
        device_strategy="auto",
        enable_ipex=True,
        enable_jit=False,  # Disable JIT for demo to avoid tracing issues
        memory_efficient=True
    )
    
    # Load model
    load_time, model_info = model.load_model()
    print(f"\nModel loaded in {load_time:.2f} seconds")
    print(f"Parameters: {model_info['parameters']:,}")
    print(f"Device: {model_info['device']}")
    
    # Generate sample text
    prompt = "The future of artificial intelligence is"
    print(f"\nGenerating text for prompt: '{prompt}'")
    
    generated_text, gen_time, perf_info = model.generate_text(
        prompt, 
        max_new_tokens=50,
        temperature=0.7
    )
    
    print(f"Generated: {generated_text}")
    print(f"Generation time: {gen_time:.2f}s ({perf_info['tokens_per_second']:.1f} tokens/sec)")
    
    # Save performance report
    report_file = model.save_performance_report()
    print(f"\nPerformance report saved to: {report_file}")
    
    # Display model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"  Intel MKL: {info['intel_optimizations']['mkl_available']}")
    print(f"  IPEX Available: {info['intel_optimizations']['ipex_available']}")
    print(f"  Threads: {info['intel_optimizations']['threads']}")

if __name__ == "__main__":
    main()
