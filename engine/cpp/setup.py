"""
Alternative build using pip/setuptools (simpler than CMake for many users).

Usage:
    cd engine/cpp
    pip install .

Or for development (editable):
    pip install -e .

This builds quoridor_cpp and installs it into the Python environment.
"""

import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "1.0.0"

ext_modules = [
    Pybind11Extension(
        "quoridor_cpp",
        ["bindings.cpp"],
        include_dirs=["."],
        define_macros=[("NDEBUG", "1")],
        cxx_std=17,
    ),
]

# Add optimization flags
for ext in ext_modules:
    if sys.platform == "win32":
        ext.extra_compile_args = ["/O2", "/EHsc"]
    else:
        ext.extra_compile_args = ["-O3", "-march=native"]

setup(
    name="quoridor_cpp",
    version=__version__,
    author="Barricade",
    description="C++ Quoridor engine with batched MCTS (pybind11)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10", "numpy"],
)
