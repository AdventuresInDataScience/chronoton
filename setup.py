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
    # MSVC: /GL enables whole-program optimisation; /LTCG is its link-time
    # counterpart. /arch:AVX2 gives SIMD on any modern x86 CPU.
    compile_args = ["/O2", "/GL", "/Gy", "/arch:AVX2", "/fp:fast"]
    link_args    = ["/LTCG"]
else:
    # GCC/Clang: -ffast-math allows aggressive FP reordering but also enables
    # -ffinite-math-only, which breaks isnan(). Restore it explicitly.
    compile_args = [
        "-O3",
        "-march=native",
        "-funroll-loops",
        "-ffast-math",
        "-fno-finite-math-only",  # keep isnan() correct; ffast-math breaks it
        "-flto",
        "-fomit-frame-pointer",
    ]
    link_args = ["-flto"]

extensions = [
    Extension(
        name="chronoton._cy_inner",
        sources=["src/chronoton/_cy_inner.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
            "nonecheck":        False,
            "overflowcheck":    False,
            "infer_types":      True,
        },
    ),
)
