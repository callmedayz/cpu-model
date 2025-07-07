#!/usr/bin/env python3
"""
Intel Model Evaluator
Comprehensive evaluation metrics, validation loops, and performance monitoring
for Intel-optimized GPT-2 models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os
import math
from typing import Dict, List, Tuple, Any, Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
from datetime import datetime
import psutil
import gc

# Import our Intel-optimized components
from intel_optimized_model import IntelOptimizedGPT2
from intel_memory_optimizer import MemoryOptimizedGPT2
from intel_data_pipeline import IntelDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelModelEvaluator:
    """Comprehensive model evaluation with Intel optimizations."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device_strategy: str = "auto",
        max_length: int = 512,
        batch_size: int = 4
    ):
        """
        Initialize Intel-optimized model evaluator.
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer (defaults to model_path)
            device_strategy: Device selection strategy
            max_length: Maximum sequence length for evaluation
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device_strategy = device_strategy
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize Intel optimizations
        self._setup_intel_optimizations()
        
        # Device selection
        self.device = self._select_device()
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_model_and_tokenizer()
        
        # Evaluation metrics storage
        self.evaluation_results = {}
        
        logger.info(f"Intel Model Evaluator initialized")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {self.device}")
    
    def _setup_intel_optimizations(self):
        """Configure Intel-specific optimizations."""
        torch.set_num_threads(4)
        
        if torch.backends.mkl.is_available():
            logger.info("Intel MKL optimizations enabled")
        
        self.ipex_available = False
        try:
            import intel_extension_for_pytorch as ipex  # type: ignore
            self.ipex_available = True
            logger.info("Intel Extension for PyTorch available")
        except ImportError:
            logger.warning("Intel Extension for PyTorch not available")
    
    def _select_device(self) -> torch.device:
        """Select optimal device based on strategy."""
        if self.device_strategy == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif self.ipex_available:
                try:
                    import intel_extension_for_pytorch as ipex
                    if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                        return torch.device("xpu")
                except:
                    pass
            return torch.device("cpu")
        else:
            return torch.device(self.device_strategy)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer for evaluation."""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Apply Intel optimizations
        if self.ipex_available and self.device.type in ["cpu", "xpu"]:
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model, dtype=torch.float32)
                logger.info("Intel Extension optimizations applied")
            except Exception as e:
                logger.warning(f"Failed to apply Intel optimizations: {e}")
        
        logger.info(f"Model loaded successfully. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def calculate_perplexity(self, dataloader: DataLoader) -> float:
        """Calculate perplexity on evaluation dataset."""
        logger.info("Calculating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Accumulate loss and token count
                batch_size, seq_len = input_ids.shape
                total_loss += loss.item() * batch_size * seq_len
                total_tokens += batch_size * seq_len
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity
    
    def evaluate_generation_quality(self, prompts: List[str], max_new_tokens: int = 50) -> Dict[str, Any]:
        """Evaluate text generation quality with various metrics."""
        logger.info("Evaluating generation quality...")
        
        generation_results = []
        generation_times = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating for prompt {i+1}/{len(prompts)}")
            
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate text
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):].strip()
            
            generation_results.append({
                'prompt': prompt,
                'generated_text': new_text,
                'full_text': generated_text,
                'generation_time': generation_time,
                'tokens_per_second': max_new_tokens / generation_time
            })
        
        # Calculate aggregate metrics
        avg_generation_time = np.mean(generation_times)
        avg_tokens_per_second = np.mean([r['tokens_per_second'] for r in generation_results])
        
        quality_metrics = {
            'generation_results': generation_results,
            'avg_generation_time': avg_generation_time,
            'avg_tokens_per_second': avg_tokens_per_second,
            'total_prompts': len(prompts)
        }
        
        logger.info(f"Average generation time: {avg_generation_time:.2f}s")
        logger.info(f"Average tokens per second: {avg_tokens_per_second:.2f}")
        
        return quality_metrics
    
    def benchmark_inference_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark inference performance with various input sizes."""
        logger.info(f"Benchmarking inference performance ({num_iterations} iterations)...")
        
        benchmark_results = {}
        
        # Test different sequence lengths
        seq_lengths = [64, 128, 256, 512]
        
        for seq_len in seq_lengths:
            if seq_len > self.max_length:
                continue
                
            logger.info(f"Testing sequence length: {seq_len}")
            
            # Create sample input
            input_ids = torch.randint(
                0, self.tokenizer.vocab_size,
                (self.batch_size, seq_len),
                device=self.device
            )
            
            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(input_ids)
            
            # Benchmark
            times = []
            memory_usage = []
            
            for i in range(num_iterations):
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
                
                # Cleanup every 10 iterations
                if i % 10 == 0:
                    gc.collect()
            
            # Calculate statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            tokens_per_second = (self.batch_size * seq_len) / avg_time
            avg_memory = np.mean(memory_usage) if memory_usage else 0
            
            benchmark_results[f'seq_len_{seq_len}'] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'tokens_per_second': tokens_per_second,
                'avg_memory_mb': avg_memory,
                'batch_size': self.batch_size,
                'sequence_length': seq_len
            }
            
            logger.info(f"Seq {seq_len}: {avg_time:.4f}s ± {std_time:.4f}s, {tokens_per_second:.1f} tokens/s")
        
        return benchmark_results
    
    def comprehensive_evaluation(
        self,
        eval_dataloader: DataLoader,
        generation_prompts: List[str],
        output_dir: str = "./evaluation_results"
    ) -> Dict[str, Any]:
        """Run comprehensive model evaluation."""
        logger.info("Starting comprehensive model evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        evaluation_start_time = time.time()
        
        # 1. Calculate perplexity
        logger.info("=== Perplexity Evaluation ===")
        perplexity = self.calculate_perplexity(eval_dataloader)
        
        # 2. Evaluate generation quality
        logger.info("=== Generation Quality Evaluation ===")
        generation_metrics = self.evaluate_generation_quality(generation_prompts)
        
        # 3. Benchmark inference performance
        logger.info("=== Inference Performance Benchmark ===")
        performance_metrics = self.benchmark_inference_performance()
        
        # 4. System information
        system_info = {
            'device': str(self.device),
            'model_path': self.model_path,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'intel_optimizations': {
                'mkl_available': torch.backends.mkl.is_available(),
                'ipex_available': self.ipex_available,
                'num_threads': torch.get_num_threads()
            },
            'system_memory_gb': psutil.virtual_memory().total / (1024**3),
            'evaluation_time': time.time() - evaluation_start_time
        }
        
        # Compile results
        comprehensive_results = {
            'perplexity': perplexity,
            'generation_metrics': generation_metrics,
            'performance_metrics': performance_metrics,
            'system_info': system_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = os.path.join(output_dir, "comprehensive_evaluation.json")
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive evaluation completed in {system_info['evaluation_time']:.2f} seconds")
        logger.info(f"Results saved to: {results_path}")
        
        return comprehensive_results


def create_evaluation_prompts() -> List[str]:
    """Create diverse prompts for generation quality evaluation."""
    return [
        "The future of artificial intelligence",
        "In a world where technology",
        "Scientists have recently discovered",
        "The most important lesson I learned",
        "Climate change is affecting",
        "The benefits of renewable energy",
        "Machine learning algorithms can",
        "The history of human civilization",
        "Space exploration has revealed",
        "The impact of social media on"
    ]


def main():
    """Demonstration of Intel model evaluation."""
    print("Intel Model Evaluator Demo")
    print("=" * 50)
    
    # Use the best model from training
    model_path = "./demo_training_output/best_model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run intel_training_loop.py first to train a model.")
        return
    
    # Initialize evaluator
    evaluator = IntelModelEvaluator(
        model_path=model_path,
        device_strategy="auto",
        max_length=128,
        batch_size=2
    )
    
    # Create evaluation data
    print("Creating evaluation data...")
    from intel_training_loop import create_sample_training_data
    eval_texts = create_sample_training_data()[:10]  # Smaller set for evaluation
    
    # Create evaluation dataloader
    data_pipeline = IntelDataPipeline(
        tokenizer_name=model_path,
        max_length=128,
        batch_size=2,
        num_workers=0
    )
    eval_dataloader = data_pipeline.create_dataloader(eval_texts, shuffle=False)
    
    # Create generation prompts
    generation_prompts = create_evaluation_prompts()
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(
        eval_dataloader=eval_dataloader,
        generation_prompts=generation_prompts,
        output_dir="./evaluation_results"
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Perplexity: {results['perplexity']:.4f}")
    print(f"Average generation time: {results['generation_metrics']['avg_generation_time']:.2f}s")
    print(f"Average tokens per second: {results['generation_metrics']['avg_tokens_per_second']:.2f}")
    print(f"Total evaluation time: {results['system_info']['evaluation_time']:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
