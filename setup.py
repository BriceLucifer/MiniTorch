from __future__ import annotations

from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup


extensions = [
    Extension(
        "MiniTorch.native._native_trainer",
        ["MiniTorch/native/_native_trainer.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
)
