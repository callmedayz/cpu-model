# AI Development Environment - Intel Optimized

This project sets up an AI development environment optimized for Intel hardware, specifically targeting the Dell Inspiron 15 5510 with Intel i5-11320H processor and Intel Iris Xe Graphics.

## Hardware Specifications

- **CPU**: Intel i5-11320H (4 cores, 8 threads, up to 4.5GHz)
- **Memory**: 16GB DDR4 (~7-8GB available for AI workloads)
- **Graphics**: Intel Iris Xe Graphics (96 execution units, 2GB shared memory)
- **Storage**: 970GB available across two NVMe SSDs

## Features

- ✅ PyTorch 2.7.1 with CPU optimizations
- ✅ Intel MKL acceleration for mathematical operations
- ✅ Transformers library for NLP models
- ✅ Jupyter environment for interactive development
- ✅ Memory-optimized configuration for 7-8GB constraint
- ✅ Multi-threading optimization for Intel i5-11320H
- ✅ TensorBoard for training monitoring

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd plant
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv ai_env
   ai_env\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Verification

Run the setup verification script to ensure everything is working correctly:

```bash
python test_intel_setup.py
```

This script tests:
- Basic PyTorch and Intel MKL setup
- Tensor operations and performance
- Intel CPU optimizations
- Memory usage and allocation
- Transformers library integration

## Performance Expectations

Based on hardware analysis and testing:

- **Training**: 2-8 tokens/second (depending on model size and batch size)
- **Inference**: 10-80 tokens/second (GPT-2 small: ~50-80 tokens/sec)
- **Memory**: Up to 7-8GB available for model and data
- **Threading**: 4 threads optimal for Intel i5-11320H

## Recommended Models

For optimal performance on this hardware:

- **GPT-2 Small** (124M parameters) - Recommended starting point
- **DistilBERT** (66M parameters) - For classification tasks
- **T5-Small** (60M parameters) - For text-to-text tasks

## Development Workflow

1. **Start Jupyter Lab**:
   ```bash
   ai_env\Scripts\activate
   jupyter lab
   ```

2. **Monitor training with TensorBoard**:
   ```bash
   tensorboard --logdir=logs
   ```

3. **Test model performance**:
   ```bash
   python test_intel_setup.py
   ```

## Memory Management

- **Available Memory**: ~7-8GB for AI workloads
- **Model Loading**: Use `torch.float16` or quantization for larger models
- **Batch Size**: Start with small batches (1-4) and increase gradually
- **Gradient Checkpointing**: Enable for memory-intensive training

## Threading Configuration

The environment is pre-configured for optimal threading:
- **PyTorch threads**: 4 (matches CPU cores)
- **OpenMP threads**: 4
- **Intel MKL**: Enabled with multi-threading

## File Structure

```
plant/
├── ai_env/                 # Virtual environment
├── test_intel_setup.py     # Setup verification script
├── requirements.txt        # Python dependencies
├── PLAN.md                # Detailed development plan
├── TODO.md                # Task breakdown
├── DEVICE_SPECS.md        # Hardware analysis
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Next Steps

1. Follow the tasks outlined in `TODO.md`
2. Review the detailed plan in `PLAN.md`
3. Start with GPT-2 small model experimentation
4. Implement memory-efficient training strategies
5. Explore Intel-specific optimizations

## Troubleshooting

- **Memory Issues**: Reduce batch size or use gradient checkpointing
- **Slow Performance**: Verify Intel MKL is enabled and threading is configured
- **Model Loading**: Use CPU-only models and avoid CUDA dependencies

## Resources

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
