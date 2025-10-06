#!/usr/bin/env python3
"""
Document Generator CLI Launcher
Simplified interface for creating test documents with embedded prompts
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_generator.core import main

if __name__ == "__main__":
    print("🚀 LLM Evaluation Document Generator")
    print("=" * 50)
    print("Creating test documents with steganographic prompt injection")
    print("=" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all dependencies are installed: pip3 install python-docx")