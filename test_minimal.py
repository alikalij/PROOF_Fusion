#!/usr/bin/env python3
"""
Minimal test script to check if PROOF project can run with basic dependencies
"""
import sys
import os

def test_imports():
    """Test if basic imports work"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import logging
        print("✅ Logging module available")
    except ImportError:
        print("❌ Logging module not found")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🔍 Testing project structure...")
    
    required_files = [
        "main.py",
        "trainer.py", 
        "models/proof.py",
        "utils/factory.py",
        "exps/cifar.json"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")
            return False
    
    return True

def test_config_loading():
    """Test if config files can be loaded"""
    print("\n🔍 Testing config loading...")
    
    try:
        import json
        with open("exps/cifar.json", "r") as f:
            config = json.load(f)
        print("✅ Config file loaded successfully")
        print(f"   - Dataset: {config.get('dataset', 'N/A')}")
        print(f"   - Model: {config.get('model_name', 'N/A')}")
        print(f"   - Device: {config.get('device', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def main():
    print("🚀 Testing PROOF project setup...")
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some basic imports failed. Please install missing packages.")
        return False
    
    # Test project structure
    if not test_project_structure():
        print("\n⚠️  Project structure incomplete. Please check file paths.")
        return False
    
    # Test config loading
    if not test_config_loading():
        print("\n⚠️  Config loading failed. Please check config files.")
        return False
    
    print("\n🎉 All tests passed! Project is ready for debugging.")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 