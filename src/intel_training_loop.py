#!/usr/bin/env python3
"""
Intel-Optimized Training Loop
Comprehensive training implementation with Intel optimizations, gradient checkpointing,
and memory management for Intel i5-11320H hardware
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
import psutil
import gc
from typing import Dict, List, Tuple, Any, Optional
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    get_linear_schedule_with_warmup
)
import logging
from datetime import datetime
import math

# Import our Intel-optimized components
from intel_optimized_model import IntelOptimizedGPT2
from intel_memory_optimizer import MemoryOptimizedGPT2, IntelMemoryOptimizer
from intel_data_pipeline import IntelDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelOptimizedTrainer:
    """Intel-optimized training loop with memory management and performance monitoring."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        save_steps: int = 100,
        eval_steps: int = 50,
        logging_steps: int = 10,
        output_dir: str = "./intel_training_output",
        enable_gradient_checkpointing: bool = True,
        target_memory_gb: float = 6.0,
        device_strategy: str = "auto"
    ):
        """
        Initialize Intel-optimized trainer.
        
        Args:
            model_name: HuggingFace model name
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            output_dir: Directory to save outputs
            enable_gradient_checkpointing: Enable gradient checkpointing
            target_memory_gb: Target memory usage in GB
            device_strategy: Device selection strategy
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.target_memory_gb = target_memory_gb
        self.device_strategy = device_strategy
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Intel optimizations
        self._setup_intel_optimizations()
        
        # Initialize components
        self.device = self._select_device()
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.memory_optimizer = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Performance tracking
        self.training_stats = {
            'losses': [],
            'learning_rates': [],
            'memory_usage': [],
            'step_times': [],
            'gradient_norms': []
        }
        
        logger.info(f"Intel-Optimized Trainer initialized")
        logger.info(f"Target memory: {target_memory_gb}GB, Device: {self.device}")
        logger.info(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    
    def _setup_intel_optimizations(self):
        """Configure Intel-specific optimizations."""
        # Set optimal threading for Intel i5-11320H
        torch.set_num_threads(4)
        
        # Enable Intel MKL optimizations
        if torch.backends.mkl.is_available():
            logger.info("Intel MKL optimizations enabled")
        
        # Try to import Intel Extension for PyTorch
        self.ipex_available = False
        try:
            import intel_extension_for_pytorch as ipex  # type: ignore
            self.ipex_available = True
            logger.info("Intel Extension for PyTorch available")
        except ImportError:
            logger.warning("Intel Extension for PyTorch not available, using standard PyTorch")
    
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
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with Intel optimizations."""
        logger.info(f"Setting up model: {self.model_name}")
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize memory optimizer
        self.memory_optimizer = IntelMemoryOptimizer(target_memory_gb=self.target_memory_gb)
        
        # Load model configuration
        config = GPT2Config.from_pretrained(self.model_name)
        
        # Create memory-optimized model
        if self.enable_gradient_checkpointing:
            self.model = MemoryOptimizedGPT2(
                config=config,
                enable_gradient_checkpointing=True,
                checkpoint_every_n_layers=2
            )
            # Load pretrained weights
            pretrained_model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.gpt2.load_state_dict(pretrained_model.state_dict())
        else:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name, config=config)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Apply Intel optimizations
        if self.ipex_available and self.device.type in ["cpu", "xpu"]:
            try:
                import intel_extension_for_pytorch as ipex
                self.model = ipex.optimize(self.model)
                logger.info("Intel Extension optimizations applied to model")
            except Exception as e:
                logger.warning(f"Failed to apply Intel Extension optimizations: {e}")
        
        # Set model to training mode
        self.model.train()
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Create learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Optimizer: AdamW with lr={self.learning_rate}")
        logger.info(f"Scheduler: Linear with {self.warmup_steps} warmup steps")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # System memory
        vm = psutil.virtual_memory()
        memory_info['system_total_gb'] = vm.total / (1024**3)
        memory_info['system_used_gb'] = vm.used / (1024**3)
        memory_info['system_available_gb'] = vm.available / (1024**3)
        
        # PyTorch memory (if CUDA)
        if torch.cuda.is_available() and self.device.type == "cuda":
            memory_info['torch_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['torch_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return memory_info
    
    def log_training_step(self, step: int, loss: float, lr: float, step_time: float, grad_norm: float):
        """Log training step information."""
        memory_info = self.get_memory_usage()
        
        # Store stats
        self.training_stats['losses'].append(loss)
        self.training_stats['learning_rates'].append(lr)
        self.training_stats['step_times'].append(step_time)
        self.training_stats['gradient_norms'].append(grad_norm)
        self.training_stats['memory_usage'].append(memory_info)
        
        # Log to console
        if step % self.logging_steps == 0:
            logger.info(
                f"Step {step}/{self.max_steps} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {step_time:.2f}s | "
                f"Grad Norm: {grad_norm:.4f} | "
                f"Memory: {memory_info['system_used_gb']:.1f}GB"
            )
    
    def save_checkpoint(self, step: int, loss: float):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        if hasattr(self.model, 'gpt2'):
            # MemoryOptimizedGPT2 case
            self.model.gpt2.save_pretrained(checkpoint_dir)
        else:
            # Standard GPT2LMHeadModel case
            self.model.save_pretrained(checkpoint_dir)
        
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'step': step,
            'epoch': self.epoch,
            'loss': loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats
        }
        
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"Checkpoint saved at step {step}")
        
        # Update best model if loss improved
        if loss < self.best_loss:
            self.best_loss = loss
            best_dir = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            
            if hasattr(self.model, 'gpt2'):
                self.model.gpt2.save_pretrained(best_dir)
            else:
                self.model.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            
            logger.info(f"New best model saved with loss: {loss:.4f}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Execute single training step."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return loss.item() * self.gradient_accumulation_steps, loss.item()

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop with Intel optimizations."""
        logger.info("Starting Intel-optimized training...")

        # Calculate total training steps
        num_training_steps = min(self.max_steps, len(train_dataloader) * 100)  # Assume max 100 epochs

        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler(num_training_steps)

        # Training loop
        self.global_step = 0
        accumulated_loss = 0.0

        start_time = time.time()

        while self.global_step < self.max_steps:
            for batch_idx, batch in enumerate(train_dataloader):
                if self.global_step >= self.max_steps:
                    break

                step_start_time = time.time()

                # Training step
                _, scaled_loss = self.train_step(batch)
                accumulated_loss += scaled_loss

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    ).item()

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Calculate average loss
                    avg_loss = accumulated_loss / self.gradient_accumulation_steps
                    accumulated_loss = 0.0

                    # Get current learning rate
                    current_lr = self.scheduler.get_last_lr()[0]

                    # Calculate step time
                    step_time = time.time() - step_start_time

                    # Log training step
                    self.log_training_step(
                        self.global_step, avg_loss, current_lr, step_time, grad_norm
                    )

                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(self.global_step, avg_loss)

                    # Evaluation
                    if eval_dataloader and self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        logger.info(f"Evaluation at step {self.global_step}: Loss = {eval_loss:.4f}")
                        self.model.train()  # Return to training mode

                    # Memory cleanup
                    if self.global_step % 50 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    self.global_step += 1

            self.epoch += 1

        # Final save
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        self.save_checkpoint(self.global_step, avg_loss)

        # Save final training report
        self.save_training_report(total_time)

    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

    def save_training_report(self, total_time: float):
        """Save comprehensive training report."""
        report = {
            'training_config': {
                'model_name': self.model_name,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_steps': self.max_steps,
                'warmup_steps': self.warmup_steps,
                'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
                'target_memory_gb': self.target_memory_gb,
                'device': str(self.device)
            },
            'training_results': {
                'total_steps': self.global_step,
                'total_epochs': self.epoch,
                'total_time_seconds': total_time,
                'final_loss': self.training_stats['losses'][-1] if self.training_stats['losses'] else None,
                'best_loss': self.best_loss,
                'avg_step_time': sum(self.training_stats['step_times']) / len(self.training_stats['step_times']) if self.training_stats['step_times'] else 0
            },
            'performance_stats': {
                'losses': self.training_stats['losses'][-100:],  # Last 100 steps
                'learning_rates': self.training_stats['learning_rates'][-100:],
                'step_times': self.training_stats['step_times'][-100:],
                'gradient_norms': self.training_stats['gradient_norms'][-100:],
                'final_memory_usage': self.training_stats['memory_usage'][-1] if self.training_stats['memory_usage'] else {}
            },
            'intel_optimizations': {
                'mkl_available': torch.backends.mkl.is_available(),
                'ipex_available': self.ipex_available,
                'num_threads': torch.get_num_threads(),
                'gradient_checkpointing': self.enable_gradient_checkpointing
            }
        }

        report_path = os.path.join(self.output_dir, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved to: {report_path}")


def create_sample_training_data() -> List[str]:
    """Create sample training data for demonstration."""
    sample_texts = [
        "The future of artificial intelligence lies in creating systems that can understand and generate human-like text with remarkable accuracy and creativity.",
        "Machine learning has revolutionized how we approach complex problems, enabling computers to learn patterns from data without explicit programming.",
        "Natural language processing combines computational linguistics with machine learning to help computers understand, interpret, and generate human language.",
        "Deep learning neural networks have achieved breakthrough results in image recognition, speech processing, and language understanding tasks.",
        "The development of large language models has opened new possibilities for automated content generation, translation, and conversational AI systems.",
        "Artificial intelligence research focuses on creating intelligent agents that can perceive their environment and take actions to maximize their success.",
        "Computer vision algorithms enable machines to identify and analyze visual content, from simple object detection to complex scene understanding.",
        "Reinforcement learning allows AI systems to learn optimal behaviors through trial and error, similar to how humans learn from experience.",
        "The intersection of AI and robotics is creating autonomous systems capable of performing complex tasks in dynamic, real-world environments.",
        "Ethical considerations in AI development include ensuring fairness, transparency, and accountability in automated decision-making systems."
    ]

    # Expand the dataset by creating variations
    expanded_texts = []
    for text in sample_texts:
        expanded_texts.append(text)
        # Add some variations
        expanded_texts.append(f"In recent years, {text.lower()}")
        expanded_texts.append(f"Researchers have found that {text.lower()}")

    return expanded_texts


def main():
    """Demonstration of Intel-optimized training."""
    print("Intel-Optimized Training Loop Demo")
    print("=" * 50)

    # Create sample training data
    print("Creating sample training data...")
    sample_texts = create_sample_training_data()
    print(f"Created {len(sample_texts)} training samples")

    # Initialize data pipeline
    data_pipeline = IntelDataPipeline(
        tokenizer_name="gpt2",
        max_length=128,
        batch_size=2,  # Small batch size for demo
        num_workers=0
    )

    # Create training dataloader
    print("Creating training dataloader...")
    train_dataloader = data_pipeline.create_dataloader(sample_texts, shuffle=True)

    # Initialize trainer
    trainer = IntelOptimizedTrainer(
        model_name="gpt2",
        learning_rate=5e-5,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=20,  # Small number for demo
        save_steps=10,
        eval_steps=10,
        logging_steps=2,
        output_dir="./models/demo_training_output",
        enable_gradient_checkpointing=True,
        target_memory_gb=6.0
    )

    # Setup model and tokenizer
    print("Setting up model and tokenizer...")
    trainer.setup_model_and_tokenizer()

    # Start training
    print("Starting training...")
    trainer.train(train_dataloader)

    print("Training completed successfully!")
    print(f"Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
