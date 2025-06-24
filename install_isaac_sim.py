#!/usr/bin/env python3
"""
Isaac Sim Installation Script for PPO Project

This script helps install Isaac Sim and its dependencies for the PPO project.
"""

import subprocess
import sys
import os
import platform


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available: {torch.cuda.get_device_name()}")
            # Access CUDA version through torch.version
            cuda_version = getattr(torch.version, 'cuda', 'Unknown')
            print(f"  CUDA version: {cuda_version}")
            return True
        else:
            print("⚠ CUDA is not available. Isaac Sim will run on CPU (slower)")
            return False
    except ImportError:
        print("⚠ PyTorch not installed. Cannot check CUDA availability")
        return False


def install_isaac_sim_packages():
    """Install Isaac Sim Python packages."""
    packages = [
        "omni-isaac-gym>=1.0.0",
        "omni-isaac-sim>=2023.1.0", 
        "omni-isaac-gym-envs>=1.0.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True


def install_additional_dependencies():
    """Install additional dependencies for Isaac Sim."""
    packages = [
        "Pillow>=8.0.0",
        "scipy>=1.7.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    return True


def test_isaac_sim_import():
    """Test if Isaac Sim can be imported."""
    print("\nTesting Isaac Sim import...")
    
    try:
        import isaacgym
        print("✓ isaacgym imported successfully")
        
        from isaacgym import gymapi
        print("✓ gymapi imported successfully")
        
        from isaacgym import gymtorch
        print("✓ gymtorch imported successfully")
        
        print("✓ Isaac Sim installation test passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Isaac Sim import failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have installed Isaac Sim from Omniverse Launcher")
        print("2. Check if your Python environment is activated")
        print("3. Try reinstalling the packages: pip install --force-reinstall omni-isaac-gym")
        return False


def main():
    """Main installation function."""
    print("=" * 60)
    print("Isaac Sim Installation Script for PPO Project")
    print("=" * 60)
    
    # Check system requirements
    print("\n1. Checking system requirements...")
    if not check_python_version():
        sys.exit(1)
    
    check_cuda()
    
    # Install Isaac Sim packages
    print("\n2. Installing Isaac Sim Python packages...")
    if not install_isaac_sim_packages():
        print("\n✗ Failed to install Isaac Sim packages")
        print("Please make sure you have Isaac Sim installed from Omniverse Launcher")
        sys.exit(1)
    
    # Install additional dependencies
    print("\n3. Installing additional dependencies...")
    if not install_additional_dependencies():
        print("\n✗ Failed to install additional dependencies")
        sys.exit(1)
    
    # Test installation
    print("\n4. Testing Isaac Sim installation...")
    if not test_isaac_sim_import():
        print("\n✗ Isaac Sim installation test failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Isaac Sim installation completed successfully!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run an Isaac Sim example:")
    print("   python examples/isaac_cartpole_example.py")
    print("2. Or run the vectorized version:")
    print("   python examples/isaac_vec_cartpole_example.py")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main() 