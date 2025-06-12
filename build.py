#!/usr/bin/env python3
"""
Reality Stone Rust Build System
Simple, fast, and reliable with maturin
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Colors for output
OK_GREEN = '\033[92m'
INFO_BLUE = '\033[94m'
WARNING_YELLOW = '\033[93m'
FAIL_RED = '\033[91m'
ENDC = '\033[0m'

def log(message, color=INFO_BLUE):
    print(f"{color}{message}{ENDC}")

def run_command(cmd, description):
    """Run a command and return success status"""
    log(f"🚀 {description}...", INFO_BLUE)
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        if result.returncode == 0:
            log(f"✅ Success: {description}", OK_GREEN)
            return True
        else:
            log(f"❌ Failed: {description} (exit code: {result.returncode})", FAIL_RED)
            return False
    except Exception as e:
        log(f"❌ Error: {e}", FAIL_RED)
        return False

def check_rust():
    """Check if Rust is installed"""
    try:
        result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            log(f"✅ Rust installed: {result.stdout.strip()}", OK_GREEN)
            return True
    except:
        pass
    
    log("❌ Rust not found!", FAIL_RED)
    log("📦 Please install Rust from https://rustup.rs/", WARNING_YELLOW)
    return False

def main():
    """Main build script"""
    log("🚀 Reality Stone Rust Build System", OK_GREEN)
    log("   Powered by Rust + PyO3 + maturin", INFO_BLUE)
    log("   No more C++ build hell! 🦀", WARNING_YELLOW)
    
    # Handle clean command
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        log("🧹 Cleaning build artifacts...", INFO_BLUE)
        dirs_to_clean = ["target", "build", "dist", "*.egg-info", "__pycache__"]
        for pattern in dirs_to_clean:
            for path in Path(".").glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        log("✅ Clean complete", OK_GREEN)
        return
    
    # Check Rust installation
    if not check_rust():
        sys.exit(1)
    
    # Install/upgrade maturin
    log("📦 Installing build tools...", INFO_BLUE)
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "maturin"], 
                      "Install maturin"):
        sys.exit(1)
    
    # Install Python dependencies
    deps = ["numpy>=1.21.0,<2.0.0", "torch>=2.0.0"]
    if not run_command([sys.executable, "-m", "pip", "install"] + deps,
                      "Install Python dependencies"):
        log("⚠️ Failed to install dependencies, continuing anyway...", WARNING_YELLOW)
    
    # Build with maturin
    log("🔨 Building Reality Stone with Rust...", INFO_BLUE)
    
    python_executable = sys.executable
    log(f"🐍 Using Python interpreter: {python_executable}", INFO_BLUE)

    build_cmd = ["maturin", "build", "--release", "--interpreter", python_executable]
    
    if "--debug" in sys.argv:
        build_cmd = ["maturin", "build", "--interpreter", python_executable]
        log("🐛 Building in debug mode", WARNING_YELLOW)
    
    if not run_command(build_cmd, "Build Rust extension wheel"):
        sys.exit(1)
        
    # Install the built wheel
    wheel_dir = Path("target/wheels")
    wheels = list(wheel_dir.glob("*.whl"))
    if not wheels:
        log("❌ No wheel found to install", FAIL_RED)
        sys.exit(1)
        
    latest_wheel = max(wheels, key=os.path.getctime)
    log(f"📦 Installing built wheel: {latest_wheel.name}", INFO_BLUE)
    if not run_command([python_executable, "-m", "pip", "install", str(latest_wheel), "--force-reinstall"], "Install wheel"):
        sys.exit(1)
    
    # Test import
    log("🧪 Testing build...", INFO_BLUE)
    test_code = """
import torch
import reality_stone
print('✅ All imports successful!')
print(f'🦀 Reality Stone (Rust Edition) loaded!')
"""
    
    if run_command([sys.executable, "-c", test_code], "Test import"):
        log("\n🎉🎉🎉 BUILD SUCCESSFUL! 🎉🎉🎉", OK_GREEN)
        log("🦀 Reality Stone is now powered by Rust!", OK_GREEN)
        log("⚡ Fast, safe, and no more build hell!", WARNING_YELLOW)
    else:
        log("\n⚠️ Build completed but import failed", WARNING_YELLOW)

if __name__ == "__main__":
    main()