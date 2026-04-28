"""
setup.py — build the chronoton._cy_inner Cython extension.

The extension is compiled as part of the package (chronoton._cy_inner) so
that it is importable via relative import from within the package.

Usage:
    # Editable install (develops in-place; Cython extension also compiled)
    pip install -e .

    # Or build the extension only, without installing:
    python setup.py build_ext --inplace

Troubleshooting:
    - Windows needs Microsoft C++ Build Tools (MSVC).
    - macOS needs Xcode Command Line Tools: xcode-select --install
    - Linux needs gcc/clang + Python dev headers: sudo apt install python3-dev
    - numpy must be installed before building: uv pip install numpy
"""

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

if sys.platform == "win32":
    compile_args = ["/O2"]
else:
    compile_args = ["-O3", "-ffunction-sections", "-fdata-sections"]

extensions = [
    Extension(
        name="chronoton._cy_inner",
        sources=["src/chronoton/_cy_inner.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
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
)
