#!/usr/bin/env python3
"""
Script to install requirements for PROOF project with fallback methods
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def install_package(package_name, pip_name=None):
    """Try different methods to install a package"""
    if pip_name is None:
        pip_name = package_name
    
    methods = [
        f"pip install {pip_name}",
        f"pip install {pip_name} --index-url https://pypi.org/simple/",
        f"pip install {pip_name} --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org"
    ]
    
    for method in methods:
        if run_command(method, f"Installing {package_name}"):
            return True
    
    print(f"⚠️  Failed to install {package_name} with all methods")
    return False

def main():
    print("🚀 Installing requirements for PROOF project...")
    
    # List of packages to install
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"), 
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("open_clip", "open-clip"),
        ("timm", "timm")
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package_name, pip_name in packages:
        if install_package(package_name, pip_name):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("🎉 All packages installed successfully!")
    else:
        print("⚠️  Some packages failed to install. You may need to install them manually.")
        print("💡 Try running: pip install <package_name> --user")

if __name__ == "__main__":
    main() 