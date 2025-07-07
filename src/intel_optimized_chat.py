#!/usr/bin/env python3
"""
Intel-Optimized Chat Script
Optimized for Intel i5-11320H with Intel MKL acceleration
Based on benchmark results showing optimal performance with GPT-2 small
"""

import torch
import time
import psutil
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import sys

class IntelOptimizedChat:
    def __init__(self, model_name="gpt2", max_length=100, temperature=0.7):
        """Initialize the Intel-optimized chat system."""
        print("🚀 Initializing Intel-Optimized Chat System...")
        
        # Set optimal threading for Intel i5-11320H
        torch.set_num_threads(4)
        print(f"   Threading: {torch.get_num_threads()} threads")
        print(f"   Intel MKL: {torch.backends.mkl.is_available()}")
        
        # Load model and tokenizer
        print(f"   Loading {model_name}...")
        start_time = time.time()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch.float32)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        load_time = time.time() - start_time
        param_count = sum(p.numel() for p in self.model.parameters())
        
        print(f"   ✅ Model loaded in {load_time:.2f} seconds")
        print(f"   📊 Parameters: {param_count:,}")
        
        # Configuration
        self.max_length = max_length
        self.temperature = temperature
        self.conversation_history = []
        
        # Performance tracking
        self.generation_times = []
        
    def get_memory_usage(self):
        """Get current memory usage."""
        memory = psutil.virtual_memory()
        return {
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent
        }
    
    def generate_response(self, user_input, max_new_tokens=50):
        """Generate a response using Intel-optimized settings."""
        
        # Prepare conversation context
        if self.conversation_history:
            # Include recent conversation history (last 3 exchanges)
            recent_history = self.conversation_history[-6:]  # 3 user + 3 assistant
            context = " ".join(recent_history) + f" Human: {user_input} Assistant:"
        else:
            context = f"Human: {user_input} Assistant:"
        
        # Tokenize input
        inputs = self.tokenizer(
            context, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        )
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )
        
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new assistant response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            response = full_response[len(context):].strip()
        
        # Clean up response
        response = response.split("Human:")[0].strip()  # Stop at next human input
        
        # Update conversation history
        self.conversation_history.append(f"Human: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        
        # Keep history manageable (last 10 exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response, generation_time
    
    def get_performance_stats(self):
        """Get performance statistics."""
        if not self.generation_times:
            return "No generations yet."
        
        avg_time = sum(self.generation_times) / len(self.generation_times)
        min_time = min(self.generation_times)
        max_time = max(self.generation_times)
        
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_generations': len(self.generation_times)
        }
    
    def interactive_chat(self):
        """Start interactive chat session."""
        print("\n" + "="*60)
        print("🤖 Intel-Optimized Chat System Ready!")
        print("="*60)
        print("Commands:")
        print("  /stats  - Show performance statistics")
        print("  /memory - Show memory usage")
        print("  /clear  - Clear conversation history")
        print("  /quit   - Exit chat")
        print("="*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == '/clear':
                    self.conversation_history = []
                    print("🧹 Conversation history cleared.")
                    continue
                elif user_input.lower() == '/stats':
                    stats = self.get_performance_stats()
                    if isinstance(stats, dict):
                        print(f"📊 Performance Stats:")
                        print(f"   Average generation time: {stats['avg_time']:.2f}s")
                        print(f"   Fastest generation: {stats['min_time']:.2f}s")
                        print(f"   Slowest generation: {stats['max_time']:.2f}s")
                        print(f"   Total generations: {stats['total_generations']}")
                    else:
                        print(stats)
                    continue
                elif user_input.lower() == '/memory':
                    memory = self.get_memory_usage()
                    print(f"💾 Memory Usage:")
                    print(f"   Available: {memory['available_gb']:.2f} GB")
                    print(f"   Used: {memory['used_percent']:.1f}%")
                    continue
                
                # Generate response
                print("🤔 Thinking...", end="", flush=True)
                response, gen_time = self.generate_response(user_input)
                
                # Clear "Thinking..." and show response
                print(f"\r🤖 Assistant: {response}")
                print(f"   ⚡ Generated in {gen_time:.2f}s ({50/gen_time:.1f} tokens/sec)")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Intel-Optimized Chat System")
    parser.add_argument("--model", default="gpt2", 
                       choices=["gpt2", "gpt2-medium", "gpt2-large"],
                       help="Model to use (default: gpt2)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature (default: 0.7)")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum context length (default: 100)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum new tokens per response (default: 50)")
    
    args = parser.parse_args()
    
    # Display system info
    print("Intel-Optimized Chat System")
    print("="*40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Intel MKL available: {torch.backends.mkl.is_available()}")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    memory = psutil.virtual_memory()
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print("="*40)
    
    try:
        # Initialize chat system
        chat = IntelOptimizedChat(
            model_name=args.model,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # Start interactive chat
        chat.interactive_chat()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error initializing chat system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
