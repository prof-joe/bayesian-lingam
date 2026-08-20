#!/usr/bin/env python3
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "_full_optimizer_backend_v4",
        ["_full_optimizer_backend_v4.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="full_optimizer_backend_v4",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
    ),
)
