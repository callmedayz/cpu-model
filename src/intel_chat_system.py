#!/usr/bin/env python3
"""
Intel-Optimized Production Chat System
Complete chat system integrating all Intel optimizations with conversation management,
performance monitoring, and production-ready features
"""

import torch
import time
import json
import os
import uuid
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import psutil
import threading
from dataclasses import dataclass, asdict
from collections import deque

# Import our Intel-optimized components
from intel_optimized_model import IntelOptimizedGPT2
from intel_memory_optimizer import MemoryOptimizedGPT2
from intel_model_evaluator import IntelModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message."""
    id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    generation_time: Optional[float] = None
    tokens_per_second: Optional[float] = None

@dataclass
class ChatSession:
    """Represents a chat session with conversation history."""
    session_id: str
    created_at: str
    last_activity: str
    messages: List[ChatMessage]
    total_tokens: int = 0
    total_generation_time: float = 0.0

class IntelChatSystem:
    """Production-ready chat system with Intel optimizations."""
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device_strategy: str = "auto",
        max_length: int = 512,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        conversation_memory: int = 10,
        enable_performance_monitoring: bool = True,
        log_conversations: bool = True,
        conversations_dir: str = "./conversations"
    ):
        """
        Initialize Intel-optimized chat system.
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
            device_strategy: Device selection strategy
            max_length: Maximum sequence length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            conversation_memory: Number of messages to keep in context
            enable_performance_monitoring: Enable performance tracking
            log_conversations: Save conversations to disk
            conversations_dir: Directory to save conversations
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device_strategy = device_strategy
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.conversation_memory = conversation_memory
        self.enable_performance_monitoring = enable_performance_monitoring
        self.log_conversations = log_conversations
        self.conversations_dir = conversations_dir
        
        # Create conversations directory
        if self.log_conversations:
            os.makedirs(conversations_dir, exist_ok=True)
        
        # Initialize Intel optimizations
        self._setup_intel_optimizations()
        
        # Device selection
        self.device = self._select_device()
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.load_model_and_tokenizer()
        
        # Session management
        self.active_sessions: Dict[str, ChatSession] = {}
        self.session_lock = threading.Lock()
        
        # Performance monitoring
        self.performance_stats = {
            'total_messages': 0,
            'total_generation_time': 0.0,
            'average_tokens_per_second': 0.0,
            'memory_usage_history': deque(maxlen=100),
            'response_times': deque(maxlen=100)
        }
        
        logger.info(f"Intel Chat System initialized")
        logger.info(f"Model: {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Max tokens: {max_new_tokens}, Temperature: {temperature}")
    
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
        """Load model and tokenizer for chat."""
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
    
    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session = ChatSession(
            session_id=session_id,
            created_at=timestamp,
            last_activity=timestamp,
            messages=[]
        )
        
        with self.session_lock:
            self.active_sessions[session_id] = session
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID."""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def build_conversation_context(self, session: ChatSession) -> str:
        """Build conversation context from recent messages."""
        # Get recent messages within memory limit
        recent_messages = session.messages[-self.conversation_memory:]
        
        # Build context string
        context_parts = []
        for message in recent_messages:
            if message.role == "user":
                context_parts.append(f"Human: {message.content}")
            else:
                context_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(context_parts)
    
    def generate_response(self, session_id: str, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """Generate response for user message."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        start_time = time.time()
        
        # Add user message to session
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat()
        )
        session.messages.append(user_msg)
        
        # Build conversation context
        context = self.build_conversation_context(session)
        prompt = f"{context}\nHuman: {user_message}\nAssistant:"
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - self.max_new_tokens
        ).to(self.device)
        
        # Generate response
        generation_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generation_time = time.time() - generation_start
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_response[len(prompt):].strip()
        
        # Clean up response (remove potential repetitions or artifacts)
        response_text = self._clean_response(response_text)
        
        # Calculate performance metrics
        tokens_generated = len(self.tokenizer.encode(response_text))
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Add assistant message to session
        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response_text,
            timestamp=datetime.now().isoformat(),
            generation_time=generation_time,
            tokens_per_second=tokens_per_second
        )
        session.messages.append(assistant_msg)
        
        # Update session stats
        session.total_tokens += tokens_generated
        session.total_generation_time += generation_time
        session.last_activity = datetime.now().isoformat()
        
        # Update performance stats
        if self.enable_performance_monitoring:
            self._update_performance_stats(generation_time, tokens_per_second)
        
        # Save conversation if enabled
        if self.log_conversations:
            self._save_conversation(session)
        
        total_time = time.time() - start_time
        
        # Performance metadata
        performance_data = {
            'generation_time': generation_time,
            'total_time': total_time,
            'tokens_generated': tokens_generated,
            'tokens_per_second': tokens_per_second,
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        logger.info(f"Generated response in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
        
        return response_text, performance_data
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove potential conversation markers that might have been generated
        markers_to_remove = ["Human:", "Assistant:", "User:", "AI:"]
        for marker in markers_to_remove:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # Limit response length to prevent overly long responses
        max_response_length = 500
        if len(response) > max_response_length:
            # Try to cut at sentence boundary
            sentences = response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= max_response_length:
                    truncated += sentence + '. '
                else:
                    break
            response = truncated.strip()
        
        return response
    
    def _update_performance_stats(self, generation_time: float, tokens_per_second: float):
        """Update performance monitoring statistics."""
        self.performance_stats['total_messages'] += 1
        self.performance_stats['total_generation_time'] += generation_time
        self.performance_stats['response_times'].append(generation_time)
        
        # Update average tokens per second
        total_messages = self.performance_stats['total_messages']
        current_avg = self.performance_stats['average_tokens_per_second']
        self.performance_stats['average_tokens_per_second'] = (
            (current_avg * (total_messages - 1) + tokens_per_second) / total_messages
        )
        
        # Update memory usage
        memory_gb = psutil.virtual_memory().used / (1024**3)
        self.performance_stats['memory_usage_history'].append(memory_gb)
    
    def _save_conversation(self, session: ChatSession):
        """Save conversation to disk."""
        filename = f"conversation_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.conversations_dir, filename)
        
        # Convert session to dictionary
        session_data = asdict(session)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        stats = self.performance_stats.copy()
        
        if stats['response_times']:
            stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])
            stats['min_response_time'] = min(stats['response_times'])
            stats['max_response_time'] = max(stats['response_times'])
        
        if stats['memory_usage_history']:
            stats['avg_memory_usage_gb'] = sum(stats['memory_usage_history']) / len(stats['memory_usage_history'])
            stats['current_memory_usage_gb'] = stats['memory_usage_history'][-1]
        
        return stats
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions."""
        current_time = datetime.now()
        sessions_to_remove = []
        
        with self.session_lock:
            for session_id, session in self.active_sessions.items():
                last_activity = datetime.fromisoformat(session.last_activity)
                age_hours = (current_time - last_activity).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")


def interactive_chat_demo():
    """Interactive chat demonstration."""
    print("Intel-Optimized Chat System Demo")
    print("=" * 50)
    
    # Use the best model from training
    model_path = "./models/demo_training_output/best_model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run intel_training_loop.py first to train a model.")
        return
    
    # Initialize chat system
    chat_system = IntelChatSystem(
        model_path=model_path,
        device_strategy="auto",
        max_length=256,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        conversation_memory=5
    )
    
    # Create session
    session_id = chat_system.create_session()
    print(f"Chat session created: {session_id}")
    print("Type 'quit' to exit, 'stats' to see performance statistics")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = chat_system.get_performance_summary()
                print(f"\nPerformance Statistics:")
                print(f"Total messages: {stats['total_messages']}")
                print(f"Average response time: {stats.get('avg_response_time', 0):.2f}s")
                print(f"Average tokens/sec: {stats['average_tokens_per_second']:.1f}")
                print(f"Current memory usage: {stats.get('current_memory_usage_gb', 0):.1f}GB")
                continue
            elif not user_input:
                continue
            
            # Generate response
            response, perf_data = chat_system.generate_response(session_id, user_input)
            
            print(f"\nAssistant: {response}")
            print(f"[Generated in {perf_data['generation_time']:.2f}s, {perf_data['tokens_per_second']:.1f} tokens/s]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nChat session ended. Thank you!")
    
    # Final performance summary
    final_stats = chat_system.get_performance_summary()
    print(f"\nFinal Performance Summary:")
    print(f"Total messages: {final_stats['total_messages']}")
    print(f"Total generation time: {final_stats['total_generation_time']:.2f}s")
    print(f"Average tokens per second: {final_stats['average_tokens_per_second']:.1f}")


if __name__ == "__main__":
    interactive_chat_demo()
