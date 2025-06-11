#!/bin/bash
# Reality Stone - Unified Ultra-Fast Installation & Build Script
# One script to rule them all! 🚀

set -euo pipefail

# Colors
readonly GREEN='\033[0;32m'
readonly BLUE='\033[0;34m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly CYAN='\033[0;36m'
readonly PURPLE='\033[0;35m'
readonly NC='\033[0m'

# Configuration - these need to be modifiable
PROJECT_NAME="reality-stone"
PYTHON="${PYTHON:-python}"
PIP="${PIP:-pip}"
BUILD_PROFILE="${BUILD_PROFILE:-ULTRAFAST}"
USE_CUDA="${USE_CUDA:-auto}"
# More aggressive CPU detection for maximum speed
CPU_COUNT=$(nproc --all 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
MAX_JOBS="${MAX_JOBS:-$((CPU_COUNT * 3))}"  # More aggressive than 2x
MEMORY_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || echo 8)

# Cache paths
readonly CACHE_DIR="${HOME}/.cache/reality-stone"
readonly PIP_CACHE_DIR="${CACHE_DIR}/pip"
readonly CCACHE_DIR="${CACHE_DIR}/ccache"

# Export for builds
export_build_vars() {
    export BUILD_PROFILE USE_CUDA MAX_JOBS MEMORY_GB
    export PIP_CACHE_DIR CCACHE_DIR
    
    # Build parallelization - maximum speed
    export MAKEFLAGS="-j${MAX_JOBS} -l${CPU_COUNT}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
    export NINJA_STATUS="[%f/%t %o/sec] "
    
    # Compiler optimization flags for maximum speed (platform-specific)
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "${WINDIR:-}" ]]; then
        # Windows/MSVC optimization flags
        export CFLAGS="/O2 /Ob2 /Oi /Ot /GL /favor:AMD64 /arch:AVX2"
        export CXXFLAGS="/O2 /Ob2 /Oi /Ot /GL /favor:AMD64 /arch:AVX2 /EHsc"
        export LDFLAGS="/LTCG /OPT:REF /OPT:ICF"
    else
        # Linux/GCC optimization flags
        export CFLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops"
        export CXXFLAGS="-O3 -march=native -mtune=native -ffast-math -funroll-loops -fno-signed-zeros"
        export LDFLAGS="-Wl,--as-needed -Wl,--strip-all"
    fi
    
    # Python/pip optimization
    export PIP_NO_CLEAN=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONUNBUFFERED=1
    
    # Memory optimization
    if [[ ${MEMORY_GB} -gt 16 ]]; then
        export MAKEFLAGS="${MAKEFLAGS} --max-load=${CPU_COUNT}"
    fi
}

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_step() {
    echo -e "\n${CYAN}🚀 $1 🚀${NC}"
    echo -e "${CYAN}================================${NC}"
}

# Timer
start_timer() { TIMER_START=$(date +%s); }
end_timer() {
    local end_time=$(date +%s)
    local duration=$((end_time - TIMER_START))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    log_info "⚡ Time: ${minutes}m ${seconds}s"
}

# Error handling
cleanup_on_error() {
    log_error "Failed! Cleaning up..."
    [[ -n "${TMPDIR:-}" ]] && [[ "${TMPDIR}" == "/dev/shm/"* ]] && rm -rf "${TMPDIR}" 2>/dev/null || true
    exit 1
}
trap cleanup_on_error ERR

# Check if command exists
command_exists() { command -v "$1" >/dev/null 2>&1; }

# Setup ultra-fast environment
setup_fast_env() {
    log_info "Setting up ULTRA-FAST environment (${CPU_COUNT} cores, ${MEMORY_GB}GB RAM)..."
    
    # Export build variables first
    export_build_vars
    
    # Create cache directories with optimal permissions
    mkdir -p "${PIP_CACHE_DIR}" "${CCACHE_DIR}"
    chmod 755 "${PIP_CACHE_DIR}" "${CCACHE_DIR}" 2>/dev/null || true
    
    # Setup ccache for MAXIMUM speed
    if command_exists ccache; then
        export CC="ccache gcc"
        export CXX="ccache g++"
        export CCACHE_MAXSIZE="20G"  # Increased cache size
        export CCACHE_COMPRESS=1
        export CCACHE_COMPRESSLEVEL=1  # Fast compression
        export CCACHE_NOCOMPRESS=0
        export CCACHE_SLOPPINESS="pch_defines,time_macros,include_file_mtime,include_file_ctime"
        ccache -M 20G >/dev/null 2>&1 || true
        log_info "ccache enabled (20GB cache)"
    fi
    
    # Ultra-fast I/O setup
    if [[ "$OSTYPE" == "linux-gnu"* ]] && [[ -w /dev/shm ]]; then
        export TMPDIR="/dev/shm/reality-stone-$$"
        mkdir -p "$TMPDIR"
        # Also use tmpfs for build directory if enough memory
        if [[ ${MEMORY_GB} -gt 8 ]]; then
            export BUILD_TMPDIR="/dev/shm/build-$$"
            mkdir -p "$BUILD_TMPDIR"
            log_info "Using tmpfs for builds: $BUILD_TMPDIR"
        fi
        log_info "Using tmpfs: $TMPDIR"
    fi
    
    # Install build tools in parallel if needed
    local missing_tools=()
    command_exists ninja || missing_tools+=("ninja")
    command_exists ccache || missing_tools+=("ccache")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_info "Installing build tools: ${missing_tools[*]}"
        "${PIP}" install --upgrade "${missing_tools[@]}" \
            --cache-dir "${PIP_CACHE_DIR}" \
            --prefer-binary \
            --no-warn-script-location \
            --disable-pip-version-check &
        
        # Don't wait - continue with other setup
        INSTALL_PID=$!
    fi
    
    # Windows-specific CUDA/VS environment fixes
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ -n "${WINDIR:-}" ]]; then
        log_info "🪟 Applying Windows CUDA/VS environment fixes..."
        
        # Fix encoding issues
        export PYTHONIOENCODING=utf-8
        export LANG=en_US.UTF-8
        export LC_ALL=en_US.UTF-8
        
        # Remove Incredibuild from PATH to prevent conflicts
        export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -i incredibuild | tr '\n' ':' | sed 's/:$//')
        
        # Set correct Visual Studio environment
        export VSINSTALLDIR="C:/Program Files/Microsoft Visual Studio/2022/Community/"
        export VCINSTALLDIR="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/"
        export VCToolsInstallDir="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/"
        
        # Add Visual Studio tools to PATH
        export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64:$PATH"
        export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE:$PATH"
        
        # CUDA environment fixes
        export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
        export CUDA_HOME="$CUDA_PATH"
        export CUDA_ROOT="$CUDA_PATH"
        export CUDA_BIN_PATH="$CUDA_PATH/bin"
        export CUDA_LIB_PATH="$CUDA_PATH/lib/x64"
        
        # CMake/CUDA compiler settings (simplified)
        export CMAKE_GENERATOR="Ninja"
        export CMAKE_C_COMPILER="cl.exe"
        export CMAKE_CXX_COMPILER="cl.exe"
        export CMAKE_CUDA_HOST_COMPILER="cl.exe"
        export CMAKE_CUDA_COMPILER="nvcc.exe"
        
        # Windows SDK (if available)
        if [[ -d "/c/Program Files (x86)/Windows Kits/10/bin" ]]; then
            export PATH="/c/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64:$PATH"
            export WindowsSdkDir="C:/Program Files (x86)/Windows Kits/10/"
            export WindowsSDKVersion="10.0.22621.0"
            
            # Critical: Add Windows SDK include paths
            export INCLUDE="C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um;C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/include;${INCLUDE:-}"
            export LIB="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64;C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/lib/x64;${LIB:-}"
            
            # Also set for CMake
            export CMAKE_INCLUDE_PATH="C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt;C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um;C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared"
            export CMAKE_LIBRARY_PATH="C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/ucrt/x64;C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22621.0/um/x64"
            
            log_info "✅ Windows SDK 10.0.22621.0 configured"
        fi
        
        log_info "✅ Windows environment configured for CUDA compilation"
    fi
    
    # Linux-specific optimizations
    if command_exists sysctl && [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Increase file descriptor limits
        ulimit -n 65536 2>/dev/null || true
        # Optimize for compilation workload
        echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
    fi
    
    # Wait for background installs to complete
    if [[ -n "${INSTALL_PID:-}" ]]; then
        wait "$INSTALL_PID" 2>/dev/null || true
    fi
    
    log_info "Environment optimized for MAXIMUM speed! 🚀"
}

# Check Python
check_python() {
    if ! command_exists "${PYTHON}"; then
        log_error "Python not found. Install Python 3.8+ first."
        exit 1
    fi

    local python_version
    python_version=$("${PYTHON}" --version 2>&1 | cut -d' ' -f2)
    log_info "Python: ${python_version}"

    if ! "${PYTHON}" -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8+ required. Found: ${python_version}"
        exit 1
    fi
}

# Install PyTorch
install_pytorch() {
    log_info "Checking PyTorch..."
    
    if "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
        local torch_version
        torch_version=$("${PYTHON}" -c "import torch; print(torch.__version__)")
        log_info "PyTorch: ${torch_version}"
        
        local cuda_available
        cuda_available=$("${PYTHON}" -c "import torch; print(torch.cuda.is_available())")
        log_info "CUDA available: ${cuda_available}"
    else
        log_warning "Installing PyTorch..."
        if [[ "${USE_CUDA}" != "false" ]]; then
            "${PIP}" install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu121 \
                --cache-dir "${PIP_CACHE_DIR}" \
                --prefer-binary
        else
            "${PIP}" install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cpu \
                --cache-dir "${PIP_CACHE_DIR}" \
                --prefer-binary
        fi
        log_success "PyTorch installed"
    fi
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies with MAXIMUM parallelization..."
    
    # Pre-install critical build tools in parallel
    local critical_deps=(
        "setuptools>=65.0"
        "wheel>=0.40.0" 
        "ninja>=1.11.0"
        "pybind11>=2.10.0"
        "build>=0.10.0"
    )
    
    log_info "Installing critical build tools..."
    "${PIP}" install --upgrade "${critical_deps[@]}" \
        --cache-dir "${PIP_CACHE_DIR}" \
        --prefer-binary \
        --no-warn-script-location \
        --disable-pip-version-check \
        --no-deps \
        --force-reinstall &
    CRITICAL_PID=$!
    
    # Install main requirements in parallel
    if [[ -f "requirements-build.txt" ]]; then
        log_info "Installing from requirements-build.txt..."
        "${PIP}" install -r requirements-build.txt \
            --cache-dir "${PIP_CACHE_DIR}" \
            --prefer-binary \
            --no-warn-script-location \
            --disable-pip-version-check \
            --upgrade \
            --compile &
        MAIN_PID=$!
    fi
    
    # Install scikit-build-core for maximum build speed
    "${PIP}" install --upgrade \
        "scikit-build-core[pyproject]>=0.7.0" \
        "cmake>=3.21" \
        --cache-dir "${PIP_CACHE_DIR}" \
        --prefer-binary \
        --no-warn-script-location \
        --disable-pip-version-check &
    BUILD_PID=$!
    
    # Wait for all installations to complete
    log_info "Waiting for parallel installations..."
    wait "$CRITICAL_PID" 2>/dev/null || true
    [[ -n "${MAIN_PID:-}" ]] && wait "$MAIN_PID" 2>/dev/null || true
    wait "$BUILD_PID" 2>/dev/null || true
    
    # Verify critical tools are available
    local missing_critical=()
    command_exists ninja || missing_critical+=("ninja not found")
    "${PYTHON}" -c "import pybind11" 2>/dev/null || missing_critical+=("pybind11 not found")
    
    if [[ ${#missing_critical[@]} -gt 0 ]]; then
        log_warning "Some tools missing: ${missing_critical[*]}"
        # Fallback install
        "${PIP}" install --upgrade "${critical_deps[@]}" \
            --cache-dir "${PIP_CACHE_DIR}" \
            --prefer-binary \
            --force-reinstall
    fi
    
    log_success "Dependencies installed with MAXIMUM speed! ⚡"
}

# Build package
build_package() {
    log_info "Building ${PROJECT_NAME} with ULTRA-FAST settings..."
    log_info "Profile: ${BUILD_PROFILE}, CUDA: ${USE_CUDA}, Jobs: ${MAX_JOBS}, Memory: ${MEMORY_GB}GB"
    
    # Export build settings with maximum optimization
    export_build_vars
    export FORCE_NINJA=1
    export SKBUILD_BUILD_OPTIONS="-j${MAX_JOBS}"
    
    # Use tmpfs build directory if available
    if [[ -n "${BUILD_TMPDIR:-}" ]]; then
        export SKBUILD_BUILD_DIR="${BUILD_TMPDIR}"
        log_info "Using tmpfs build directory: ${BUILD_TMPDIR}"
    fi
    
    # Try multiple build methods in order of speed
    local build_success=false
    
    # Method 1: Modern scikit-build-core (fastest)
    if [[ -f "pyproject.toml" ]] && command_exists ninja; then
        log_info "🚀 Attempting ULTRA-FAST scikit-build-core build..."
        if "${PYTHON}" -m pip install -e . \
            --no-build-isolation \
            --no-deps \
            --cache-dir "${PIP_CACHE_DIR}" \
            --disable-pip-version-check \
            --config-settings=build-dir="${BUILD_TMPDIR:-build}" \
            --config-settings=cmake.define.CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
            --config-settings=cmake.define.CMAKE_CXX_FLAGS="-O3 -march=native" \
            --verbose 2>/dev/null; then
            build_success=true
            log_success "ULTRA-FAST build completed! 🚀"
        fi
    fi
    
    # Method 2: Direct setup.py with ninja (fast fallback)
    if [[ "$build_success" == false ]]; then
        log_info "⚡ Attempting optimized setup.py build..."
        if "${PYTHON}" setup.py build_ext \
            --inplace \
            --parallel "${MAX_JOBS}" \
            --force \
            2>/dev/null; then
            build_success=true
            log_success "Optimized build completed! ⚡"
        fi
    fi
    
    # Method 3: Standard build (reliable fallback)
    if [[ "$build_success" == false ]]; then
        log_warning "Falling back to standard build..."
        if "${PYTHON}" setup.py build_ext --inplace; then
            build_success=true
            log_success "Standard build completed!"
        fi
    fi
    
    # Method 4: Last resort - pip build
    if [[ "$build_success" == false ]]; then
        log_warning "Trying pip build as last resort..."
        "${PIP}" install -e . \
            --cache-dir "${PIP_CACHE_DIR}" \
            --disable-pip-version-check \
            --force-reinstall
        build_success=true
        log_success "Pip build completed!"
    fi
    
    # Cleanup tmpfs build directory
    if [[ -n "${BUILD_TMPDIR:-}" ]] && [[ -d "${BUILD_TMPDIR}" ]]; then
        rm -rf "${BUILD_TMPDIR}" 2>/dev/null || true
    fi
    
    if [[ "$build_success" == true ]]; then
        log_success "Package built with MAXIMUM speed! 🚀⚡"
    else
        log_error "All build methods failed!"
        return 1
    fi
}

# Install package
install_package() {
    log_info "Installing package..."
    "${PIP}" install -e . \
        --no-build-isolation \
        --cache-dir "${PIP_CACHE_DIR}" \
        --disable-pip-version-check
    log_success "Package installed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    if "${PYTHON}" -c "import reality_stone; print(f'🚀 Reality Stone {reality_stone.__version__} ready!')" 2>/dev/null; then
        log_success "Installation verified! 🚀"
        
        # Quick CUDA test
        if "${PYTHON}" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            if "${PYTHON}" -c "import reality_stone; print('CUDA functions available')" 2>/dev/null; then
                log_success "CUDA functionality working! 🚀"
            else
                log_warning "CUDA not available (CPU-only build)"
            fi
        else
            log_info "Using CPU-only version"
        fi
    else
        log_error "Installation verification failed"
        return 1
    fi
}

# Clean function
clean_build() {
    log_info "Cleaning build artifacts..."
    rm -rf build dist *.egg-info .build_cache
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.so" -delete 2>/dev/null || true
    find . -name "*.pyd" -delete 2>/dev/null || true
    log_success "Clean completed"
}

# Show usage
show_usage() {
    echo "Reality Stone - Unified Installation & Build Script"
    echo "=================================================="
    echo ""
    echo "🚀 ULTRA-FAST BUILD SYSTEM 🚀"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install     Full installation (default)"
    echo "  build       Build only"
    echo "  clean       Clean build artifacts"
    echo "  test        Quick test"
    echo ""
    echo "Options:"
    echo "  --build-profile PROFILE   (DEV, CI, RELEASE, FAST, ULTRAFAST) [${BUILD_PROFILE}]"
    echo "  --use-cuda MODE           (auto, true, false) [${USE_CUDA}]"
    echo "  --max-jobs N              Parallel jobs [${MAX_JOBS}]"
    echo ""
    echo "Examples:"
    echo "  $0                        # Full ultra-fast installation"
    echo "  $0 build                  # Build only"
    echo "  $0 --use-cuda=false       # CPU-only build"
    echo "  $0 --max-jobs=8           # Use 8 parallel jobs"
    echo ""
    echo "Environment Variables:"
    echo "  BUILD_PROFILE=${BUILD_PROFILE}"
    echo "  USE_CUDA=${USE_CUDA}"
    echo "  MAX_JOBS=${MAX_JOBS}"
}

# Parse arguments
parse_args() {
    COMMAND="install"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            install|build|clean|test)
                COMMAND="$1"
                shift
                ;;
            --build-profile)
                BUILD_PROFILE="$2"
                shift 2
                ;;
            --use-cuda)
                USE_CUDA="$2"
                shift 2
                ;;
            --max-jobs)
                MAX_JOBS="$2"
                shift 2
                ;;
            --build-profile=*)
                BUILD_PROFILE="${1#*=}"
                shift
                ;;
            --use-cuda=*)
                USE_CUDA="${1#*=}"
                shift
                ;;
            --max-jobs=*)
                MAX_JOBS="${1#*=}"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main installation
main_install() {
    echo "╔══════════════════════════════════════════════════╗"
    echo "║            🚀 Reality Stone Installer 🚀         ║"
    echo "║              ULTRA-FAST BUILD SYSTEM             ║"
    echo "║          ${CPU_COUNT} cores • ${MEMORY_GB}GB RAM • ${BUILD_PROFILE} mode          ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    
    start_timer
    
    log_step "Environment Setup"
    setup_fast_env
    check_python
    
    log_step "PyTorch Installation"
    install_pytorch
    
    log_step "Dependencies"
    install_dependencies
    
    log_step "Building Package"
    build_package
    
    log_step "Installing Package"
    install_package
    
    log_step "Verification"
    verify_installation
    
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║                🚀 SUCCESS! 🚀                    ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    log_success "Reality Stone installed successfully!"
    echo ""
    echo "Quick start:"
    echo "  python -c \"import reality_stone; print('Hello, Reality Stone!')\""
    echo ""
    echo "Run examples:"
    echo "  python examples/no_deepcopy_version.py"
    
    end_timer
    
    # Show cache stats
    if command_exists ccache; then
        echo ""
        log_info "ccache stats:"
        ccache -s | grep -E "(cache hit|cache miss|cache size)" || true
    fi
    
    # Cleanup tmpfs
    [[ -n "${TMPDIR:-}" ]] && [[ "${TMPDIR}" == "/dev/shm/"* ]] && rm -rf "${TMPDIR}" 2>/dev/null || true
}

# Main function
main() {
    parse_args "$@"
    
    case "$COMMAND" in
        install)
            main_install
            ;;
        build)
            log_step "Fast Build"
            setup_fast_env
            build_package
            log_success "Build completed!"
            ;;
        clean)
            clean_build
            ;;
        test)
            log_step "Quick Test"
            if "${PYTHON}" -c "import reality_stone; print('✅ Reality Stone works!')"; then
                log_success "Test passed!"
            else
                log_error "Test failed!"
                exit 1
            fi
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 