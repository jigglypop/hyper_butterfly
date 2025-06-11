#!/usr/bin/env python3
"""
Ultra-fast Build Script for Reality Stone
Optimized with scikit-build-core + CMake + Ninja
"""
import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

OK_GREEN = '\033[92m'
INFO_BLUE = '\033[94m'
WARNING_YELLOW = '\033[93m'
FAIL_RED = '\033[91m'
ENDC = '\033[0m'

def log(message, color=INFO_BLUE):
    print(f"{color}{message}{ENDC}")

def find_msvc_compiler():
    """Automatically find the latest MSVC compiler without using vcvars64.bat"""
    if platform.system() != "Windows":
        return os.environ
    
    try:
        log("🔍 Auto-detecting MSVC compiler...", INFO_BLUE)
        
        # Find Visual Studio installation
        vs_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional", 
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise"
        ]
        
        vs_path = None
        for path in vs_paths:
            if Path(path).exists():
                vs_path = Path(path)
                log(f"  ✅ Found Visual Studio: {path}", OK_GREEN)
                break
        
        if not vs_path:
            raise FileNotFoundError("Visual Studio not found")
        
        # Find latest MSVC version
        msvc_base = vs_path / "VC/Tools/MSVC"
        if not msvc_base.exists():
            raise FileNotFoundError(f"MSVC tools not found in {msvc_base}")
        
        # Get all MSVC versions and sort to find latest
        msvc_versions = [d.name for d in msvc_base.iterdir() if d.is_dir()]
        if not msvc_versions:
            raise FileNotFoundError("No MSVC versions found")
        
        # Sort versions to get the latest
        latest_version = sorted(msvc_versions)[-1]
        compiler_dir = msvc_base / latest_version / "bin/HostX64/x64"
        compiler_path = compiler_dir / "cl.exe"
        
        if not compiler_path.exists():
            raise FileNotFoundError(f"Compiler not found: {compiler_path}")
        
        log(f"  ✅ Found compiler: {compiler_path}", OK_GREEN)
        
        # Find Windows SDK
        sdk_paths = [
            "C:/Program Files (x86)/Windows Kits/10",
            "C:/Program Files/Windows Kits/10"
        ]
        
        sdk_path = None
        for path in sdk_paths:
            if Path(path).exists():
                sdk_path = Path(path)
                break
        
        if not sdk_path:
            log("  ⚠️ Windows SDK not found, trying fallback", WARNING_YELLOW)
        else:
            log(f"  ✅ Found Windows SDK: {sdk_path}", OK_GREEN)
        
        # Build environment without vcvars64.bat
        env = os.environ.copy()
        
        # Set compiler paths
        env.update({
            "CC": str(compiler_path),
            "CXX": str(compiler_path),
            "CMAKE_C_COMPILER": str(compiler_path),
            "CMAKE_CXX_COMPILER": str(compiler_path),
            "CMAKE_GENERATOR": "Ninja",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_BUILD_PARALLEL_LEVEL": "8",  # 병렬 빌드
            "PYTHONIOENCODING": "utf-8",
        })
        
        # Set up include and library paths
        if sdk_path:
            # Find latest SDK version for includes and libs
            sdk_include = sdk_path / "Include"
            sdk_lib = sdk_path / "Lib"
            
            if sdk_include.exists() and sdk_lib.exists():
                sdk_versions = [d.name for d in sdk_include.iterdir() if d.is_dir() and d.name.startswith("10.")]
                if sdk_versions:
                    latest_sdk = sorted(sdk_versions)[-1]
                    
                    # Include paths
                    include_paths = [
                        str(sdk_include / latest_sdk / "ucrt"),
                        str(sdk_include / latest_sdk / "um"),
                        str(sdk_include / latest_sdk / "shared"),
                        str(vs_path / f"VC/Tools/MSVC/{latest_version}/include"),
                    ]
                    
                    # Library paths
                    lib_paths = [
                        str(sdk_lib / latest_sdk / "ucrt/x64"),
                        str(sdk_lib / latest_sdk / "um/x64"),
                        str(vs_path / f"VC/Tools/MSVC/{latest_version}/lib/x64"),
                    ]
                    
                    # Set environment variables
                    env["INCLUDE"] = ";".join(include_paths)
                    env["LIB"] = ";".join(lib_paths)
                    
                    log(f"  ✅ Added SDK include and lib paths for {latest_sdk}", OK_GREEN)
        
        # Add essential paths
        essential_paths = [
            str(compiler_dir),  # Compiler directory
            "C:/Program Files/CMake/bin",  # CMake
            str(vs_path / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja"),  # Ninja
            str(vs_path / "Common7/Tools"),  # VS Tools
            "C:/Windows/System32",
            str(Path(sys.executable).parent),  # Python
        ]
        if sdk_path:
            sdk_bin = sdk_path / "bin"
            if sdk_bin.exists():
                sdk_versions = [d.name for d in sdk_bin.iterdir() if d.is_dir() and d.name.startswith("10.")]
                if sdk_versions:
                    latest_sdk = sorted(sdk_versions)[-1]
                    essential_paths.extend([
                        str(sdk_bin / latest_sdk / "x64"),  # RC.exe and other tools
                        str(sdk_bin / latest_sdk / "x86"),  # Fallback tools
                    ])
                    log(f"  ✅ Added Windows SDK {latest_sdk} tools", OK_GREEN)
        # Add more VS paths for completeness
        essential_paths.extend([
            str(vs_path / "VC/bin"),
            str(vs_path / "Common7/IDE"),
            str(vs_path / "MSBuild/Current/Bin"),
        ])
        
        # Filter existing paths and clean PATH
        clean_paths = [p for p in essential_paths if Path(p).exists()]
        
        # Remove any incredibuild paths to avoid conflicts
        original_path = env.get("PATH", "")
        filtered_paths = [p for p in original_path.split(";") if "incredibuild" not in p.lower()]
        
        # Combine clean paths with filtered original paths
        env["PATH"] = ";".join(clean_paths + filtered_paths)
        
        log("  ✅ Complete Visual Studio environment configured", OK_GREEN)
        return env
        
    except Exception as e:
        log(f"  ❌ Failed to find MSVC compiler: {e}", FAIL_RED)
        log("  Please ensure Visual Studio with C++ workload is installed", WARNING_YELLOW)
        return None

def run_command(command_list, description, env):
    """Runs a command with proper encoding and error handling"""
    log(f"🚀 Executing: {description}...", INFO_BLUE)
    try:
        process = subprocess.Popen(
            command_list,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore'  # Ignore encoding errors
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip())
        
        if process.returncode == 0:
            log(f"✅ Success: {description}", OK_GREEN)
            return True
        else:
            log(f"❌ Failure: {description} (exit code: {process.returncode})", FAIL_RED)
            return False
            
    except Exception as e:
        log(f"❌ Exception in {description}: {e}", FAIL_RED)
        return False

def main():
    """Main build script logic"""
    log("🚀 Reality Stone Ultra-Fast Build System", OK_GREEN)
    log("   Using: scikit-build-core + CMake + Ninja + Parallel Compilation", INFO_BLUE)
    
    # Handle clean command
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        log("🧹 Cleaning build artifacts...", INFO_BLUE)
        shutil.rmtree("build", ignore_errors=True)
        shutil.rmtree("dist", ignore_errors=True)
        for path in Path(".").rglob("*.egg-info"):
            shutil.rmtree(path, ignore_errors=True)
        log("✅ Clean complete", OK_GREEN)
        return

    # Auto-detect compiler environment
    build_env = find_msvc_compiler()
    if platform.system() == "Windows" and not build_env:
        log("❌ Cannot proceed without compiler", FAIL_RED)
        sys.exit(1)

    # Install build dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "wheel", "setuptools"], "Install build essentials", build_env):
        sys.exit(1)

    # Fix NumPy installation conflicts
    log("🔧 Fixing NumPy installation conflicts...", INFO_BLUE)
    if not run_command([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], "Remove conflicting NumPy", build_env):
        log("⚠️ NumPy uninstall failed, continuing...", WARNING_YELLOW)
    
    if not run_command([sys.executable, "-m", "pip", "install", "numpy>=1.21.0,<2.0.0", "--force-reinstall"], "Install compatible NumPy", build_env):
        sys.exit(1)

    # Main build with scikit-build-core (ultra fast)
    build_cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--verbose", "--no-build-isolation"]
    if not run_command(build_cmd, "Build Reality Stone with C++ extensions", build_env):
        sys.exit(1)

    # Test the build
    log("\n🧪 Testing build...", INFO_BLUE)
    test_cmd = [sys.executable, "-c", "import reality_stone; import reality_stone._C; print('✅ All imports successful!')"]
    if run_command(test_cmd, "Test import", build_env):
        log("\n🎉🎉🎉 ULTRA-FAST BUILD SUCCESSFUL! 🎉🎉🎉", OK_GREEN)
        log("🚀 Reality Stone is ready to use!", INFO_BLUE)
        log("⚡ Built with Ninja + Parallel Compilation for maximum speed!", WARNING_YELLOW)
    else:
        log("\n⚠️ Build completed but import test failed", WARNING_YELLOW)
        log("You may need to check your Python environment", WARNING_YELLOW)

if __name__ == "__main__":
    main()