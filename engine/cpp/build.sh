#!/bin/bash
# Build script for the C++ Quoridor engine with pybind11 bindings.
#
# Prerequisites:
#   pip install pybind11 numpy
#   CMake >= 3.16
#   A C++17 compiler (g++, clang++, or MSVC)
#
# Usage:
#   cd engine/cpp
#   chmod +x build.sh
#   ./build.sh
#
# The built module (quoridor_cpp.so or quoridor_cpp.pyd) will be copied
# to the engine/ directory so Python can import it directly.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Building C++ Quoridor Engine ==="
echo "Source: $SCRIPT_DIR"
echo "Output: $ENGINE_DIR"

# Ensure pybind11 is installed
python3 -c "import pybind11" 2>/dev/null || python -c "import pybind11" 2>/dev/null || {
    echo "Installing pybind11..."
    pip install pybind11
}

# Create build directory
mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

# Configure
echo ""
echo "--- Configuring with CMake ---"
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "--- Compiling ---"
if command -v nproc &> /dev/null; then
    JOBS=$(nproc)
elif command -v sysctl &> /dev/null; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi
cmake --build . --config Release -j "$JOBS"

# Copy the built module to the engine directory
echo ""
echo "--- Installing ---"
# Find the built module (.so on Linux/Mac, .pyd on Windows)
MODULE=$(find . -name "quoridor_cpp*" \( -name "*.so" -o -name "*.pyd" \) | head -1)

if [ -z "$MODULE" ]; then
    echo "ERROR: Could not find built module!"
    exit 1
fi

cp "$MODULE" "$ENGINE_DIR/"
echo "Copied $(basename "$MODULE") to $ENGINE_DIR/"

echo ""
echo "=== Build complete! ==="
echo ""
echo "Test it:"
echo "  cd $ENGINE_DIR"
echo "  python -c \"import quoridor_cpp; g = quoridor_cpp.QuoridorGame(); print(g)\""
