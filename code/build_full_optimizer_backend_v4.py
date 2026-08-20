#!/usr/bin/env python3
from __future__ import annotations
import importlib
import importlib.machinery
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
subprocess.run(
    [sys.executable, str(HERE / "setup_full_optimizer_backend_v4.py"),
     "build_ext", "--inplace", "--force"],
    cwd=HERE,
    check=True,
)
importlib.invalidate_caches()
sys.path.insert(0, str(HERE))
module = importlib.import_module("_full_optimizer_backend_v4")
path = str(Path(module.__file__).resolve())
if not any(path.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
    raise RuntimeError(f"Backend was not loaded from a compiled extension: {path}")
print("Compiled backend:", path)
print(module.backend_info())
print("BUILD AND IMPORT VALIDATION PASSED")
