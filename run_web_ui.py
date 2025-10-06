#!/usr/bin/env python3
"""
Launcher script for the LLM Evaluation Web UI
Run this script to start the web interface
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Check if required dependencies are installed
try:
    import flask
    import flask_cors
    print("✓ Flask dependencies found")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Installing required packages...")
    os.system("pip3 install flask flask-cors werkzeug")

# Import and run the Flask app
from src.web.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting LLM Evaluation Web UI")
    print("=" * 60)
    print(f"📁 Project directory: {Path(__file__).parent}")
    print(f"🌐 Web interface will be available at: http://localhost:8001")
    print(f"📊 Results page: http://localhost:8001/results")
    print(f"📄 Document Generator: http://localhost:8001/documents")
    print("=" * 60)
    print("Features available:")
    print("• Evaluation Configuration Builder")
    print("• Results Viewer and Analytics")
    print("• Document Generator with Multi-Injection Support")
    print("• Bulk Document Generation")
    print("• Sample Templates")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8001)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("Make sure all required dependencies are installed and API keys are configured.")