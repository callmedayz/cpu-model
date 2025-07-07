#!/usr/bin/env python3
"""
Intel Hardware Benchmarking Script
Comprehensive benchmarking for Intel i5-11320H with Intel MKL optimizations
"""

import torch
import time
import psutil
import gc
import json
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import numpy as np

def print_separator(title):
    """Print a formatted separator with title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def get_memory_info():
    """Get current memory usage information."""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'percent': memory.percent
    }

def benchmark_model_loading(model_names):
    """Benchmark model loading times for different model sizes."""
    print_separator("MODEL LOADING BENCHMARK")
    
    results = {}
    
    for model_name in model_names:
        print(f"\nTesting {model_name}...")
        
        try:
            # Clear memory before each test
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            memory_before = get_memory_info()
            
            # Time model loading
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            load_time = time.time() - start_time
            
            memory_after = get_memory_info()
            
            # Get model info
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            results[model_name] = {
                'load_time': load_time,
                'parameters': param_count,
                'model_size_mb': model_size_mb,
                'memory_before_gb': memory_before['available_gb'],
                'memory_after_gb': memory_after['available_gb'],
                'memory_used_gb': memory_before['available_gb'] - memory_after['available_gb'],
                'success': True
            }
            
            print(f"✅ Loaded successfully")
            print(f"   Parameters: {param_count:,}")
            print(f"   Model size: {model_size_mb:.1f} MB")
            print(f"   Load time: {load_time:.2f} seconds")
            print(f"   Memory used: {results[model_name]['memory_used_gb']:.2f} GB")
            
            # Clean up
            del model, tokenizer
            gc.collect()
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            results[model_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def benchmark_inference_performance(model_name, batch_sizes=[1, 2, 4], sequence_lengths=[50, 100, 200]):
    """Benchmark inference performance with different configurations."""
    print_separator(f"INFERENCE BENCHMARK - {model_name}")
    
    try:
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                print(f"\nTesting batch_size={batch_size}, sequence_length={seq_len}")
                
                try:
                    # Create dummy input
                    input_text = "The future of artificial intelligence " * (seq_len // 8)
                    inputs = tokenizer(
                        [input_text] * batch_size, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=seq_len
                    )
                    
                    # Warm up
                    with torch.no_grad():
                        for _ in range(3):
                            _ = model(**inputs)
                    
                    # Benchmark
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(10):
                            outputs = model(**inputs)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 10
                    tokens_per_second = (batch_size * seq_len) / avg_time
                    
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'avg_time': avg_time,
                        'tokens_per_second': tokens_per_second,
                        'success': True
                    }
                    
                    print(f"   Average time: {avg_time:.4f} seconds")
                    print(f"   Tokens/second: {tokens_per_second:.1f}")
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results[key] = {
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'success': False,
                        'error': str(e)
                    }
        
        # Clean up
        del model, tokenizer
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to benchmark {model_name}: {e}")
        return {'error': str(e)}

def benchmark_text_generation(model_name, generation_lengths=[50, 100, 200]):
    """Benchmark text generation performance."""
    print_separator(f"TEXT GENERATION BENCHMARK - {model_name}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        results = {}
        prompt = "The future of artificial intelligence is"
        
        for gen_length in generation_lengths:
            print(f"\nTesting generation length: {gen_length} tokens")
            
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Warm up
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Benchmark
                start_time = time.time()
                
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=gen_length,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                
                generation_time = end_time - start_time
                tokens_per_second = gen_length / generation_time
                
                # Decode generated text
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                
                results[f"gen_{gen_length}"] = {
                    'generation_length': gen_length,
                    'generation_time': generation_time,
                    'tokens_per_second': tokens_per_second,
                    'generated_text': generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                    'success': True
                }
                
                print(f"   Generation time: {generation_time:.2f} seconds")
                print(f"   Tokens/second: {tokens_per_second:.1f}")
                print(f"   Generated: {generated_text[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[f"gen_{gen_length}"] = {
                    'generation_length': gen_length,
                    'success': False,
                    'error': str(e)
                }
        
        # Clean up
        del model, tokenizer
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to benchmark generation for {model_name}: {e}")
        return {'error': str(e)}

def benchmark_threading_performance():
    """Benchmark performance with different thread configurations."""
    print_separator("THREADING PERFORMANCE BENCHMARK")

    thread_counts = [1, 2, 4, 8]
    results = {}

    for num_threads in thread_counts:
        print(f"\nTesting with {num_threads} threads...")

        torch.set_num_threads(num_threads)
        # Note: set_num_interop_threads can only be called once, so we skip it
        
        # Matrix multiplication benchmark
        size = 2000
        iterations = 5
        
        times = []
        for _ in range(iterations):
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            
            start_time = time.time()
            c = torch.mm(a, b)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[num_threads] = {
            'threads': num_threads,
            'avg_time': avg_time,
            'std_time': std_time,
            'operations_per_second': 1.0 / avg_time
        }
        
        print(f"   Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
        print(f"   Operations/second: {1.0/avg_time:.2f}")
    
    # Reset to optimal
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    
    return results

def save_benchmark_results(results, filename="./results/benchmark_results.json"):
    """Save benchmark results to JSON file."""
    # Ensure results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()
    results['system_info'] = {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'pytorch_version': torch.__version__,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads()
    }

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📊 Results saved to {filename}")

def main():
    """Main benchmarking function."""
    print("Intel Hardware Benchmarking Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Intel MKL available: {torch.backends.mkl.is_available()}")
    print(f"Threads: {torch.get_num_threads()}")
    
    # Set optimal threading for Intel i5-11320H
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    
    all_results = {}
    
    # 1. Model loading benchmark
    model_names = [
        "gpt2",           # 124M parameters
        "gpt2-medium",    # 355M parameters  
        "gpt2-large",     # 774M parameters
    ]
    
    all_results['model_loading'] = benchmark_model_loading(model_names)
    
    # 2. Inference performance benchmark (only for models that loaded successfully)
    successful_models = [name for name, result in all_results['model_loading'].items() 
                        if result.get('success', False)]
    
    all_results['inference_performance'] = {}
    for model_name in successful_models[:2]:  # Test first 2 successful models
        all_results['inference_performance'][model_name] = benchmark_inference_performance(model_name)
    
    # 3. Text generation benchmark
    all_results['text_generation'] = {}
    for model_name in successful_models[:1]:  # Test first successful model
        all_results['text_generation'][model_name] = benchmark_text_generation(model_name)
    
    # 4. Threading performance benchmark
    all_results['threading_performance'] = benchmark_threading_performance()
    
    # Save results
    save_benchmark_results(all_results)
    
    print_separator("BENCHMARK COMPLETE")
    print("🎉 All benchmarks completed successfully!")
    print("📊 Results saved to benchmark_results.json")
    print("\nNext steps:")
    print("1. Review benchmark_results.json for detailed metrics")
    print("2. Identify optimal configurations for your hardware")
    print("3. Proceed to model implementation phase")

if __name__ == "__main__":
    main()
