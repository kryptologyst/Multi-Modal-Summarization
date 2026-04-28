#!/usr/bin/env python3
"""Quick start script for multi-modal summarization project."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def main():
    """Main function for quick start."""
    parser = argparse.ArgumentParser(description="Quick start for multi-modal summarization")
    parser.add_argument("--demo", action="store_true", help="Launch Streamlit demo")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--example", action="store_true", help="Run simple example")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--all", action="store_true", help="Run all checks and demo")
    
    args = parser.parse_args()
    
    print("🚀 Multi-Modal Summarization - Quick Start")
    print("=" * 50)
    
    if args.install or args.all:
        # Install dependencies
        success = run_command(
            "pip install -r requirements.txt",
            "Installing dependencies"
        )
        if not success:
            print("❌ Failed to install dependencies. Please check your Python environment.")
            return
    
    if args.test or args.all:
        # Run tests
        success = run_command(
            "python -m pytest tests/ -v",
            "Running tests"
        )
        if not success:
            print("⚠️  Some tests failed. This might be due to missing dependencies.")
    
    if args.example or args.all:
        # Run simple example
        success = run_command(
            "python 0937.py",
            "Running simple example"
        )
        if not success:
            print("⚠️  Example failed. This might be due to missing models or dependencies.")
    
    if args.demo or args.all:
        # Launch demo
        print(f"\n{'='*50}")
        print("🎯 Launching Streamlit Demo")
        print(f"{'='*50}")
        print("The demo will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the demo")
        
        try:
            subprocess.run(["streamlit", "run", "demo/streamlit_app.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to launch Streamlit demo")
            print("Make sure Streamlit is installed: pip install streamlit")
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
    
    if not any([args.demo, args.test, args.example, args.install, args.all]):
        # Show help
        print("""
📋 Available Options:

  --install    Install dependencies
  --test       Run tests  
  --example    Run simple example
  --demo       Launch Streamlit demo
  --all        Run all checks and launch demo

📚 Quick Start Guide:

1. Install dependencies:
   python quick_start.py --install

2. Run simple example:
   python quick_start.py --example

3. Launch interactive demo:
   python quick_start.py --demo

4. Run everything:
   python quick_start.py --all

🔗 For more information, see README.md
        """)


if __name__ == "__main__":
    main()
