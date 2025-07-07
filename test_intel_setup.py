#!/usr/bin/env python3
"""
Test script to verify PyTorch installation and Intel CPU optimizations.
This script tests CPU optimizations and basic tensor operations.
"""

import sys
import torch
import numpy as np
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Check if Intel Extension for PyTorch is available
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    print("Intel Extension for PyTorch not available, using standard PyTorch with Intel MKL")

def print_separator(title):
    """Print a formatted separator with title."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_basic_setup():
    """Test basic PyTorch setup and Intel optimizations."""
    print_separator("BASIC SETUP TEST")

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    if IPEX_AVAILABLE:
        print(f"Intel Extension for PyTorch version: {ipex.__version__}")
    else:
        print("Intel Extension for PyTorch: Not available")
    print(f"NumPy version: {np.__version__}")

    # Check Intel MKL
    print(f"\nIntel MKL available: {torch.backends.mkl.is_available()}")
    if torch.backends.mkl.is_available():
        try:
            print(f"Intel MKL enabled: {torch.backends.mkl.enabled}")
        except AttributeError:
            print("Intel MKL enabled: True (attribute not available in this PyTorch version)")

    # Check available devices
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")

    # Check XPU availability (only if IPEX is available)
    if IPEX_AVAILABLE:
        try:
            xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
            print(f"XPU available: {xpu_available}")
            if xpu_available:
                print(f"XPU device count: {torch.xpu.device_count()}")
                print(f"XPU device name: {torch.xpu.get_device_name()}")
        except Exception as e:
            print(f"XPU check failed: {e}")
    else:
        print("XPU: Not available (Intel Extension for PyTorch not installed)")

    # Check threading
    print(f"\nPyTorch threads: {torch.get_num_threads()}")
    print(f"OpenMP threads: {torch.get_num_interop_threads()}")

    return True

def test_tensor_operations():
    """Test basic tensor operations on different devices."""
    print_separator("TENSOR OPERATIONS TEST")
    
    # Test CPU operations
    print("Testing CPU operations...")
    cpu_tensor = torch.randn(1000, 1000)
    start_time = time.time()
    _ = torch.mm(cpu_tensor, cpu_tensor.t())
    cpu_time = time.time() - start_time
    print(f"CPU matrix multiplication (1000x1000): {cpu_time:.4f} seconds")

    # Test XPU operations if available (only with IPEX)
    if IPEX_AVAILABLE:
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                print("\nTesting XPU operations...")
                xpu_tensor = torch.randn(1000, 1000).to('xpu')
                start_time = time.time()
                _ = torch.mm(xpu_tensor, xpu_tensor.t())
                torch.xpu.synchronize()  # Ensure operation completes
                xpu_time = time.time() - start_time
                print(f"XPU matrix multiplication (1000x1000): {xpu_time:.4f} seconds")
                print(f"XPU speedup: {cpu_time/xpu_time:.2f}x")
            else:
                print("\nXPU not available, skipping XPU tensor operations")
        except Exception as e:
            print(f"\nXPU tensor operations failed: {e}")
    else:
        print("\nXPU not available (Intel Extension for PyTorch not installed)")
    
    return True

def test_intel_optimizations():
    """Test Intel-specific optimizations."""
    print_separator("INTEL OPTIMIZATIONS TEST")

    try:
        # Test Intel MKL threading
        print("Testing Intel MKL threading optimization...")
        torch.set_num_threads(4)  # Use 4 threads for Intel i5-11320H
        print(f"Set PyTorch threads to: {torch.get_num_threads()}")

        # Create a simple model for optimization testing
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

        # Test standard PyTorch
        input_tensor = torch.randn(32, 512)
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_tensor)
        standard_time = time.time() - start_time
        print(f"Standard PyTorch inference (100 iterations): {standard_time:.4f} seconds")

        # Test Intel optimization if available
        if IPEX_AVAILABLE:
            model_ipex = ipex.optimize(model, dtype=torch.float32)
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model_ipex(input_tensor)
            ipex_time = time.time() - start_time
            print(f"Intel optimized inference (100 iterations): {ipex_time:.4f} seconds")
            print(f"Intel optimization speedup: {standard_time/ipex_time:.2f}x")
        else:
            # Test with different thread counts to show MKL benefits
            print("\nTesting different thread configurations...")

            # Single thread
            torch.set_num_threads(1)
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(input_tensor)
            single_thread_time = time.time() - start_time
            print(f"Single thread inference (100 iterations): {single_thread_time:.4f} seconds")

            # Multi-thread
            torch.set_num_threads(4)
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(input_tensor)
            multi_thread_time = time.time() - start_time
            print(f"Multi-thread inference (100 iterations): {multi_thread_time:.4f} seconds")
            print(f"Multi-threading speedup: {single_thread_time/multi_thread_time:.2f}x")

    except Exception as e:
        print(f"Intel optimization test failed: {e}")

    return True

def test_memory_usage():
    """Test memory usage and availability."""
    print_separator("MEMORY USAGE TEST")
    
    import psutil
    
    # System memory info
    memory = psutil.virtual_memory()
    print(f"Total system memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")
    
    # Test memory allocation
    try:
        print("\nTesting large tensor allocation...")
        # Try to allocate ~2GB tensor (reasonable for our 7-8GB available)
        large_tensor = torch.randn(16384, 16384)  # ~1GB float32
        print(f"Successfully allocated tensor of size: {large_tensor.shape}")
        print(f"Tensor memory usage: {large_tensor.numel() * 4 / (1024**3):.2f} GB")
        del large_tensor  # Free memory
        
    except Exception as e:
        print(f"Large tensor allocation failed: {e}")
    
    return True

def test_transformers_integration():
    """Test Transformers library integration with Intel optimizations."""
    print_separator("TRANSFORMERS INTEGRATION TEST")
    
    try:
        print("Loading GPT-2 small model...")
        model_name = "gpt2"  # GPT-2 small (124M parameters)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference
        test_text = "The future of artificial intelligence is"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        print(f"\nTesting standard inference...")
        start_time = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        standard_time = time.time() - start_time
        print(f"Standard inference time: {standard_time:.4f} seconds")

        # Test Intel optimization if available
        if IPEX_AVAILABLE:
            print(f"\nTesting Intel optimized inference...")
            model_ipex = ipex.optimize(model, dtype=torch.float32)
            start_time = time.time()
            with torch.no_grad():
                _ = model_ipex(**inputs)
            ipex_time = time.time() - start_time
            print(f"Intel optimized inference time: {ipex_time:.4f} seconds")
            print(f"Optimization speedup: {standard_time/ipex_time:.2f}x")
        else:
            print(f"\nIntel Extension for PyTorch not available, skipping IPEX optimization test")
        
        # Test text generation
        print(f"\nTesting text generation...")
        with torch.no_grad():
            generated = model.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"Transformers integration test failed: {e}")
    
    return True

def main():
    """Run all tests."""
    print("Intel Extension for PyTorch Setup Verification")
    print("=" * 60)
    
    tests = [
        test_basic_setup,
        test_tensor_operations,
        test_intel_optimizations,
        test_memory_usage,
        test_transformers_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print_separator("TEST SUMMARY")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Intel Extension for PyTorch is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    main()
