"""
setup.py — build the _cy_inner Cython extension.

Usage (in the project root, alongside _cy_inner.pyx):

    # One-time install of build requirements
    pip install cython numpy

    # Build the extension in-place (produces _cy_inner.<platform>.so / .pyd)
    python setup.py build_ext --inplace

    # Or if you prefer pip (reads pyproject.toml if present):
    pip install -e .

After building, the dispatcher `cython_backtester` will find and use the
compiled module automatically. If the build fails or the extension isn't
compiled, `cython_backtester` silently falls back to the pure-Python
loop — importing it still works but `cython_available()` returns False.

Troubleshooting:
    - Windows needs Microsoft Build Tools for C++ (or a matching MSVC).
    - macOS needs Xcode Command Line Tools (`xcode-select --install`).
    - Linux needs gcc or clang, plus Python development headers
      (`python3-dev` on Debian/Ubuntu).
    - If numpy headers aren't found, confirm numpy is installed in the
      active environment; `numpy.get_include()` must resolve at build time.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="_cy_inner",
        sources=["_cy_inner.pyx"],
        include_dirs=[np.get_include()],
        # -O3 / /O2 for aggressive optimisation on the hot loop.
        # -ffast-math is intentionally NOT used: it breaks IEEE NaN/inf
        # semantics, and the loop relies on isnan() for SL/TP/TS guards.
        extra_compile_args=[
            "-O3",
            "-ffunction-sections",
            "-fdata-sections",
        ],
        # Suppress numpy deprecation warning about the old C API
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]


setup(
    name="backtester_cython_ext",
    description="Compiled inner loop for the single-file backtester.",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck":      False,
            "wraparound":       False,
            "cdivision":        True,
            "initializedcheck": False,
        },
    ),
    zip_safe=False,
)
