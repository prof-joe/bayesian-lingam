#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Google Colab runner for p=5, nu=5: standardized shape vs two LiNGAMs.

The output prefix is deliberately fixed (no timestamp) so rerunning with the
same master seed resumes completed rows.  The default target is 100 replications.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from google.colab import drive


def ask_int(prompt: str, default: int, minimum: int = 1) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    value = default if raw == "" else int(raw)
    if value < minimum:
        raise ValueError(f"{prompt} must be at least {minimum}.")
    return value


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    root = Path(
        "/content/full_optimizer_bayesian_ica_v12_p5_standardized_direct_ica_df5"
    )
    if not root.exists():
        raise FileNotFoundError(
            f"{root} がありません。先にZIPを /content に解凍してください。"
        )

    drive.mount("/content/drive", force_remount=False)

    run_command([
        sys.executable, "-m", "pip", "install", "-q",
        "cython", "setuptools", "numpy", "scipy", "lingam", "threadpoolctl",
    ])
    run_command([sys.executable, str(root / "build_full_optimizer_backend_v4.py")])

    p = 5
    df = 5.0
    reps = ask_int("target total number of replications", 100)
    seed = ask_int("master seed", 20260718, minimum=0)
    jobs = ask_int("number of worker processes", 2)

    output_dir = Path("/content/drive/MyDrive/BayesianICA/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / (
        f"p5_seed{seed}_df5_standardized_direct_ica_v12"
    )
    raw_path = Path(str(output_prefix) + "_raw.csv")

    print("\nV12 p=5, ν=5 EXPERIMENT", flush=True)
    print(f"p={p}, n=(50, 100, 200), target reps={reps}, df={df}", flush=True)
    print(f"master_seed={seed}, jobs={jobs}", flush=True)
    print(
        "methods=['proposed_standardized_shape', 'direct_lingam', 'ica_lingam']",
        flush=True,
    )
    print(f"output_prefix={output_prefix}", flush=True)
    if raw_path.exists():
        print(
            "既存のraw CSVを検出しました。完了済みの行を飛ばして再開します。",
            flush=True,
        )
    else:
        print("新規実行です。", flush=True)
    print(flush=True)

    command = [
        sys.executable,
        "-u",
        str(root / "run_p5_standardized_direct_ica_v12.py"),
        "--p-values", "5",
        "--reps", str(reps),
        "--jobs", str(jobs),
        "--parallel-unit", "condition",
        "--methods",
        "proposed_standardized_shape",
        "direct_lingam",
        "ica_lingam",
        "--beta-values", "0.4",
        "--df", "5.0",
        "--seed", str(seed),
        "--output-prefix", str(output_prefix),
    ]
    # Deliberately no --overwrite: this preserves and resumes existing rows.
    run_command(command)
    print(f"[ALL DONE] output_prefix={output_prefix}", flush=True)


if __name__ == "__main__":
    main()
