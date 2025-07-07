#!/usr/bin/env python3
"""
Intel-Optimized AI Model - Main Entry Point

This script provides a command-line interface to run different components
of the Intel-optimized AI model system.

Usage:
    python main.py chat                    # Start interactive chat system
    python main.py train                   # Run training pipeline
    python main.py evaluate                # Run model evaluation
    python main.py benchmark               # Run performance benchmarks
    python main.py test                    # Run system tests
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    parser = argparse.ArgumentParser(
        description="Intel-Optimized AI Model - Main Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py chat                 # Start interactive chat
  python main.py train                # Train the model
  python main.py evaluate             # Evaluate model performance
  python main.py benchmark            # Run benchmarks
  python main.py test                 # Run tests
        """
    )
    
    parser.add_argument(
        'command',
        choices=['chat', 'train', 'evaluate', 'benchmark', 'test'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == 'chat':
            from intel_chat_system import main as chat_main
            print("🚀 Starting Intel-Optimized Chat System...")
            chat_main()
            
        elif args.command == 'train':
            from intel_training_loop import main as train_main
            print("🏋️ Starting Intel-Optimized Training...")
            train_main()
            
        elif args.command == 'evaluate':
            from intel_model_evaluator import main as eval_main
            print("📊 Starting Model Evaluation...")
            eval_main()
            
        elif args.command == 'benchmark':
            from intel_optimized_model import main as benchmark_main
            print("⚡ Running Performance Benchmarks...")
            benchmark_main()
            
        elif args.command == 'test':
            import subprocess
            print("🧪 Running System Tests...")
            test_files = [
                "tests/test_intel_setup.py"
            ]
            for test_file in test_files:
                if os.path.exists(test_file):
                    print(f"Running {test_file}...")
                    subprocess.run([sys.executable, test_file])
                else:
                    print(f"Test file {test_file} not found")
                    
    except ImportError as e:
        print(f"❌ Error importing module: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
