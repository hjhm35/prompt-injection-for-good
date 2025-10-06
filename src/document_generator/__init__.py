"""
Document Generator Module for Prompt Injection For Good Evaluation System

This module provides tools for creating test documents with embedded prompts
using steganographic header injection techniques. The documents can be used
for LLM evaluation testing with hidden prompt metadata.
"""

from .core import DocumentGenerator, DocumentConfig, DocumentGeneratorCLI

__all__ = ['DocumentGenerator', 'DocumentConfig', 'DocumentGeneratorCLI']

__version__ = '1.0.0'