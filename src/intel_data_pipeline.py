#!/usr/bin/env python3
"""
Intel-Optimized Data Pipeline
Efficient data loading, preprocessing, and tokenization for Intel i5-11320H
Optimized for memory constraints and CPU performance
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import time
from typing import Dict, List, Tuple, Any, Iterator
from transformers import GPT2Tokenizer
import logging
from datasets import load_dataset, Dataset as HFDataset
import psutil
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelOptimizedTextDataset(Dataset):
    """Memory-efficient text dataset optimized for Intel hardware."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
        stride: int = 256,
        return_tensors: str = "pt"
    ):
        """
        Initialize dataset with Intel optimizations.
        
        Args:
            texts: List of text strings
            tokenizer: GPT2 tokenizer
            max_length: Maximum sequence length
            stride: Stride for overlapping sequences
            return_tensors: Format for returned tensors
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.return_tensors = return_tensors
        
        # Pre-tokenize and create examples
        logger.info(f"Tokenizing {len(texts)} texts...")
        self.examples = self._create_examples()
        logger.info(f"Created {len(self.examples)} training examples")
    
    def _create_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Create tokenized examples with memory optimization."""
        examples = []

        for i, text in enumerate(self.texts):
            if i % 100 == 0:
                logger.info(f"Processing text {i}/{len(self.texts)}")

            # Tokenize text
            tokens = self.tokenizer(
                text,
                return_tensors=self.return_tensors,
                truncation=False,
                padding=False
            )

            input_ids = tokens['input_ids'][0]

            # If text is shorter than max_length, pad or use as-is
            if len(input_ids) < self.max_length:
                if len(input_ids) >= 10:  # Minimum viable sequence length
                    # Pad to max_length
                    padded_ids = torch.cat([
                        input_ids,
                        torch.full((self.max_length - len(input_ids),), self.tokenizer.pad_token_id)
                    ])

                    # Create input and target (shifted by 1)
                    example = {
                        'input_ids': padded_ids[:-1].clone(),
                        'labels': padded_ids[1:].clone()
                    }
                    examples.append(example)
            else:
                # Create overlapping sequences for longer texts
                for start_idx in range(0, len(input_ids) - self.max_length + 1, self.stride):
                    end_idx = start_idx + self.max_length

                    sequence = input_ids[start_idx:end_idx]

                    # Create input and target (shifted by 1)
                    example = {
                        'input_ids': sequence[:-1].clone(),
                        'labels': sequence[1:].clone()
                    }

                    examples.append(example)

        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

class IntelDataPipeline:
    """Intel-optimized data pipeline for efficient training."""
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        max_length: int = 512,
        batch_size: int = 4,
        num_workers: int = 2,
        prefetch_factor: int = 2
    ):
        """
        Initialize Intel-optimized data pipeline.
        
        Args:
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            prefetch_factor: Number of batches to prefetch
        """
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Performance tracking
        self.loading_times = []
        self.memory_usage = []
        
        logger.info(f"Intel Data Pipeline initialized")
        logger.info(f"  Max length: {max_length}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Workers: {num_workers}")
    
    def load_dataset_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        max_samples: int = None
    ) -> List[str]:
        """Load dataset from HuggingFace Hub."""
        logger.info(f"Loading dataset: {dataset_name} ({split})")
        
        start_time = time.time()
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        # Extract texts
        texts = []
        for i, example in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            if text_column in example:
                text = example[text_column]
                if isinstance(text, str) and len(text.strip()) > 0:
                    texts.append(text.strip())
        
        load_time = time.time() - start_time
        self.loading_times.append(load_time)
        
        logger.info(f"Loaded {len(texts)} texts in {load_time:.2f} seconds")
        return texts
    
    def load_dataset_from_files(
        self,
        file_paths: List[str],
        file_format: str = "txt"
    ) -> List[str]:
        """Load dataset from local files."""
        logger.info(f"Loading {len(file_paths)} files ({file_format} format)")
        
        start_time = time.time()
        texts = []
        
        for file_path in file_paths:
            try:
                if file_format == "txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
                
                elif file_format == "jsonl":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                texts.append(data['text'])
                
                elif file_format == "json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and 'text' in item:
                                    texts.append(item['text'])
                                elif isinstance(item, str):
                                    texts.append(item)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        load_time = time.time() - start_time
        self.loading_times.append(load_time)
        
        logger.info(f"Loaded {len(texts)} texts in {load_time:.2f} seconds")
        return texts
    
    def _collate_fn(self, batch):
        """Custom collate function for efficient batching."""
        # Stack tensors efficiently
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    def create_dataloader(
        self,
        texts: List[str],
        shuffle: bool = True,
        drop_last: bool = True
    ) -> DataLoader:
        """Create optimized DataLoader for Intel hardware."""
        logger.info("Creating Intel-optimized DataLoader...")

        # Create dataset
        dataset = IntelOptimizedTextDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

        # Create DataLoader with Intel optimizations
        # Use num_workers=0 to avoid multiprocessing issues on Windows
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Disable multiprocessing for Windows compatibility
            drop_last=drop_last,
            collate_fn=self._collate_fn,
            pin_memory=False,  # CPU-only, no need for pinned memory
        )

        logger.info(f"DataLoader created with {len(dataset)} examples")
        return dataloader
    
    def validate_data_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Validate data quality and provide statistics."""
        logger.info("Validating data quality...")
        
        stats = {
            'total_texts': len(texts),
            'empty_texts': 0,
            'short_texts': 0,
            'long_texts': 0,
            'avg_length': 0,
            'min_length': float('inf'),
            'max_length': 0,
            'total_tokens': 0,
            'unique_texts': 0
        }
        
        lengths = []
        unique_texts = set()
        
        for text in texts:
            if not text or len(text.strip()) == 0:
                stats['empty_texts'] += 1
                continue
            
            length = len(text)
            lengths.append(length)
            unique_texts.add(text)
            
            stats['min_length'] = min(stats['min_length'], length)
            stats['max_length'] = max(stats['max_length'], length)
            
            if length < 50:
                stats['short_texts'] += 1
            elif length > 2000:
                stats['long_texts'] += 1
            
            # Estimate tokens (rough approximation)
            stats['total_tokens'] += len(text.split())
        
        if lengths:
            stats['avg_length'] = sum(lengths) / len(lengths)
        else:
            stats['min_length'] = 0
        
        stats['unique_texts'] = len(unique_texts)
        
        # Log statistics
        logger.info(f"Data Quality Report:")
        logger.info(f"  Total texts: {stats['total_texts']}")
        logger.info(f"  Unique texts: {stats['unique_texts']}")
        logger.info(f"  Empty texts: {stats['empty_texts']}")
        logger.info(f"  Short texts (<50 chars): {stats['short_texts']}")
        logger.info(f"  Long texts (>2000 chars): {stats['long_texts']}")
        logger.info(f"  Average length: {stats['avg_length']:.1f} characters")
        logger.info(f"  Length range: {stats['min_length']} - {stats['max_length']}")
        logger.info(f"  Estimated tokens: {stats['total_tokens']:,}")
        
        return stats
    
    def benchmark_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, Any]:
        """Benchmark DataLoader performance."""
        logger.info(f"Benchmarking DataLoader for {num_batches} batches...")
        
        batch_times = []
        memory_usage = []
        
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Simulate processing
            _ = batch['input_ids'].shape
            _ = batch['labels'].shape
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Track memory
            memory = psutil.virtual_memory()
            memory_usage.append(memory.used / (1024**3))
            
            if i % 5 == 0:
                logger.info(f"  Batch {i}: {batch_time:.4f}s")
        
        total_time = time.time() - start_time
        
        results = {
            'total_time': total_time,
            'avg_batch_time': sum(batch_times) / len(batch_times),
            'min_batch_time': min(batch_times),
            'max_batch_time': max(batch_times),
            'batches_per_second': len(batch_times) / total_time,
            'avg_memory_gb': sum(memory_usage) / len(memory_usage),
            'peak_memory_gb': max(memory_usage)
        }
        
        logger.info(f"DataLoader Benchmark Results:")
        logger.info(f"  Average batch time: {results['avg_batch_time']:.4f}s")
        logger.info(f"  Batches per second: {results['batches_per_second']:.2f}")
        logger.info(f"  Peak memory usage: {results['peak_memory_gb']:.2f}GB")
        
        return results
    
    def save_pipeline_config(self, filename: str = "./results/data_pipeline_config.json"):
        """Save pipeline configuration and performance stats."""
        # Ensure results directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        config = {
            'tokenizer_name': self.tokenizer_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor,
            'performance_stats': {
                'loading_times': self.loading_times,
                'avg_loading_time': sum(self.loading_times) / len(self.loading_times) if self.loading_times else 0
            }
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Pipeline configuration saved to {filename}")
        return filename

def main():
    """Demonstration of Intel-optimized data pipeline."""
    print("Intel-Optimized Data Pipeline Demo")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = IntelDataPipeline(
        max_length=128,  # Smaller for demo
        batch_size=4,
        num_workers=2
    )
    
    # Create sample data (longer texts for proper tokenization)
    sample_texts = [
        "The future of artificial intelligence is bright and full of possibilities. " +
        "Machine learning models are becoming more efficient and powerful every day. " +
        "Natural language processing has revolutionized how we interact with computers. " +
        "Deep learning architectures continue to evolve and improve performance. " +
        "Intel processors provide excellent performance for AI workloads and training. " +
        "CPU-based training is becoming more viable with modern optimizations and techniques.",

        "Memory efficiency is crucial for training large language models on consumer hardware. " +
        "Data preprocessing is a critical step in the machine learning pipeline process. " +
        "Gradient checkpointing helps reduce memory usage during backpropagation steps. " +
        "Batch size optimization is essential for maximizing throughput on Intel processors. " +
        "Intel MKL provides significant performance improvements for mathematical operations. " +
        "Modern transformers can be efficiently trained on CPU with proper optimizations.",

        "Text generation models have transformed natural language processing applications. " +
        "GPT-2 and similar architectures demonstrate impressive language understanding capabilities. " +
        "Fine-tuning pre-trained models is often more efficient than training from scratch. " +
        "Data quality and preprocessing significantly impact model performance and accuracy. " +
        "Intel Extension for PyTorch provides additional optimizations for Intel hardware. " +
        "Memory-mapped datasets can help manage large training corpora efficiently."
    ] * 5  # Repeat for more data
    
    print(f"Created {len(sample_texts)} sample texts")
    
    # Validate data quality
    _ = pipeline.validate_data_quality(sample_texts)

    # Create DataLoader
    dataloader = pipeline.create_dataloader(sample_texts)

    # Benchmark performance
    _ = pipeline.benchmark_dataloader(dataloader, num_batches=5)
    
    # Save configuration
    config_file = pipeline.save_pipeline_config()
    print(f"Configuration saved to: {config_file}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
