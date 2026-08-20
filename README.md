# Bayesian LiNGAM

This repository contains the source code and scripts used for the
numerical experiments in the Bayesian LiNGAM paper.

## Contents

- `code/run_p5_standardized_direct_ica_v12.py`: main experiment runner
- `code/_full_optimizer_backend_v4.pyx`: compiled backend for the proposed method
- `code/build_full_optimizer_backend_v4.py`: build and validation script
- `code/colab_p5_df5_standardized_direct_ica_runner.py`: Google Colab runner for the \(p=5\) experiment
- `code/README_P5_DF5_JA.txt`: Japanese instructions for running the \(p=5\) experiment

The main experiment runner supports

- \(p=5\), with \(n=50,100,200\)
- \(p=10\), with \(n=100,200,400\)
- \(p=15\), with \(n=150,300,600\)

Although the filename of the main runner contains `p5`, the same
program supports all three dimensions through the `--p-values`
option.

## Methods

The simulation program compares the following methods:

- Bayesian LiNGAM (`proposed_standardized_shape`)
- Direct-LiNGAM
- ICA-LiNGAM

The simulation includes the no-confounding condition and adjacent and
nonadjacent confounding conditions.

## Requirements

The code was developed and tested with Python 3.11.

Install the required Python packages from the root directory of the
repository:

```bash
pip install -r requirements.txt
```

The required packages are:

- NumPy
- SciPy
- Cython
- setuptools
- lingam
- threadpoolctl

## Building the compiled backend

Before running the numerical experiments, compile and validate the
Cython backend:

```bash
cd code
python build_full_optimizer_backend_v4.py
```

A successful build ends with the message:

```text
BUILD AND IMPORT VALIDATION PASSED
```

## Quick test

A small one-replication test can be run as follows:

```bash
python run_p5_standardized_direct_ica_v12.py --p-values 5 --reps 1 --jobs 1 --output-prefix smoke_test
```

This command is intended only to verify that the program runs
successfully.

## Reproducing the numerical experiments

The following command runs the experiments for \(p=5\), \(p=10\),
and \(p=15\):

```bash
python run_p5_standardized_direct_ica_v12.py --p-values 5 10 15 --reps 100 --jobs 2 --parallel-unit condition --methods proposed_standardized_shape direct_lingam ica_lingam --beta-values 0.4 --df 5.0 --seed 20260718 --output-prefix bayesian_lingam_v12
```

The full experiment is computationally intensive. The dimensions may
also be run separately by changing the `--p-values` option.

For example, the \(p=5\) experiment can be run by:

```bash
python run_p5_standardized_direct_ica_v12.py --p-values 5 --reps 100 --jobs 2 --parallel-unit condition --methods proposed_standardized_shape direct_lingam ica_lingam --beta-values 0.4 --df 5.0 --seed 20260718 --output-prefix p5_seed20260718_df5_standardized_direct_ica_v12
```

When the same output prefix and random seed are used again, completed
simulation rows are retained and the remaining computations are
resumed. Use the `--overwrite` option only when the existing results
should be replaced.

## Output files

For a specified output prefix, the program creates the following
files:

```text
<output-prefix>_raw.csv
<output-prefix>_summary.csv
<output-prefix>_overall_summary.csv
<output-prefix>_paired_comparisons.csv
<output-prefix>_config.json
```

## License

This software is released under the MIT License. See
[LICENSE](LICENSE) for details.

## Software archive

Version 1.0.0 of the source code and scripts used for the numerical
experiments is permanently archived on Zenodo:

https://doi.org/10.5281/zenodo.22036228

## Citation

If you use this software, please cite:

Joe Suzuki, “Bayesian ICA for Causal Discovery,” arXiv:2601.11815,
2026.

https://doi.org/10.48550/arXiv.2601.11815

Machine-readable citation metadata are provided in
[CITATION.cff](CITATION.cff).
