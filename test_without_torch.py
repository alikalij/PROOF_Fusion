#!/usr/bin/env python3
"""
Test script that can run without PyTorch to verify project structure
"""
import sys
import os
import json

def test_project_structure():
    """Test if project structure is correct"""
    print("🔍 Testing project structure...")
    
    required_files = [
        "main.py",
        "trainer.py", 
        "models/proof.py",
        "utils/factory.py",
        "exps/cifar.json"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def test_config_loading():
    """Test if config files can be loaded"""
    print("\n🔍 Testing config loading...")
    
    try:
        with open("exps/cifar.json", "r") as f:
            config = json.load(f)
        print("✅ Config file loaded successfully")
        print(f"   - Dataset: {config.get('dataset', 'N/A')}")
        print(f"   - Model: {config.get('model_name', 'N/A')}")
        print(f"   - Device: {config.get('device', 'N/A')}")
        print(f"   - Init classes: {config.get('init_cls', 'N/A')}")
        print(f"   - Increment: {config.get('increment', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def test_basic_imports():
    """Test basic Python imports"""
    print("\n🔍 Testing basic imports...")
    
    try:
        import logging
        print("✅ Logging module available")
    except ImportError:
        print("❌ Logging module not found")
        return False
    
    try:
        import copy
        print("✅ Copy module available")
    except ImportError:
        print("❌ Copy module not found")
        return False
    
    try:
        import json
        print("✅ JSON module available")
    except ImportError:
        print("❌ JSON module not found")
        return False
    
    return True

def test_optional_imports():
    """Test optional imports (PyTorch, NumPy, etc.)"""
    print("\n🔍 Testing optional imports...")
    
    optional_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib")
    ]
    
    available_packages = []
    for module_name, display_name in optional_packages:
        try:
            __import__(module_name)
            print(f"✅ {display_name} available")
            available_packages.append(display_name)
        except ImportError:
            print(f"❌ {display_name} not available")
    
    return available_packages

def main():
    print("🚀 Testing PROOF project setup (without PyTorch)...")
    
    # Test project structure
    if not test_project_structure():
        print("\n⚠️  Project structure incomplete. Please check file paths.")
        return False
    
    # Test config loading
    if not test_config_loading():
        print("\n⚠️  Config loading failed. Please check config files.")
        return False
    
    # Test basic imports
    if not test_basic_imports():
        print("\n⚠️  Basic imports failed.")
        return False
    
    # Test optional imports
    available_packages = test_optional_imports()
    
    print(f"\n📊 Summary:")
    print(f"✅ Project structure: OK")
    print(f"✅ Config loading: OK")
    print(f"✅ Basic imports: OK")
    print(f"📦 Available packages: {', '.join(available_packages) if available_packages else 'None'}")
    
    if "PyTorch" in available_packages:
        print("\n🎉 All tests passed! Project is ready for debugging.")
    else:
        print("\n⚠️  PyTorch not available. You can still test project structure.")
        print("💡 To install PyTorch, try the installation scripts.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 