# Open4DFlow — Analysis Pipeline

This folder contains the post-processing pipeline for reconstructing and extracting velocity, displacement and strain data from 4D flow MRI acquisitions.

## Pipeline Scripts

- **`reconstruct_and_process.py`** – Reconstructs velocity fields from raw k-space data, converting from ISMRMRD format to BART for image reconstruction and velocity calculation.
- **`geometry.py`** – Extracts geometry information from ISMRMRD data to compute the affine matrix required for NIfTI export.
- **`calc_strain.py`** – Computes 3D displacement fields from velocity data and calculates the full strain tensor via Savitzky-Golay spatial derivatives, outputting principal strain eigenvalues and eigenvectors in [ORMIR-MIDS](https://github.com/ORMIR-MIDS/ORMIR_MIDS) NIfTI format.

## Installation

```bash
pip install -r requirements.txt
```

For BART toolbox installation, follow the instructions at the [BART website](https://mrirecon.github.io/bart/).

## Notes

These scripts are intended for research purposes.
