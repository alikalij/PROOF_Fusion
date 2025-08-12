#!/usr/bin/env python3
"""
Install packages using Iranian mirrors and alternative sources
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

def install_with_iranian_mirrors():
    """Install packages using Iranian mirrors"""
    print("🚀 Installing packages using Iranian mirrors...")
    
    # Iranian mirrors and alternative sources
    mirrors = [
        "https://pypi.org/simple/",
        "https://pypi.anaconda.org/anaconda/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",  # Tsinghua mirror
        "https://mirrors.aliyun.com/pypi/simple/",    # Alibaba mirror
    ]
    
    packages = [
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
    ]
    
    success_count = 0
    total_packages = len(packages)
    
    for package_name, pip_name in packages:
        for mirror in mirrors:
            command = f"pip install {pip_name} --index-url {mirror} --trusted-host {mirror.replace('https://', '').replace('/simple/', '')}"
            if run_command(command, f"Installing {package_name} from {mirror}"):
                success_count += 1
                break
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{total_packages} packages")
    
    return success_count == total_packages

def install_torch_alternative():
    """Try alternative methods for PyTorch"""
    print("\n🔄 Trying alternative PyTorch installation methods...")
    
    methods = [
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org",
        "pip install torch torchvision --index-url https://pypi.org/simple/ --trusted-host pypi.org",
        "pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org",
        "pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org"
    ]
    
    for method in methods:
        if run_command(method, "Installing PyTorch"):
            return True
    
    return False

def main():
    print("🚀 Installing packages with Iranian mirrors...")
    
    # Install basic packages
    if install_with_iranian_mirrors():
        print("✅ Basic packages installed successfully!")
    else:
        print("⚠️  Some basic packages failed to install")
    
    # Install PyTorch
    if install_torch_alternative():
        print("✅ PyTorch installed successfully!")
    else:
        print("❌ PyTorch installation failed")
        print("💡 You may need to download PyTorch manually")
    
    print("\n🎉 Installation process completed!")

if __name__ == "__main__":
    main() 