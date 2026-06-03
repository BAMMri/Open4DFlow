# Open4DFlow — Analysis Pipeline

This folder contains the post-processing pipeline for reconstructing and extracting velocity, displacement and strain data from 4D flow MRI acquisitions.

A demo notebook with example data is available: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KA2dQRTU1l7U4Darnd9UKPh657nCDPp5#scrollTo=Q05HA9ElTfg6) → data is fetched automatically from [Zenodo](https://zenodo.org/records/20085845).

## Pipeline Scripts

- **`reconstruct_and_process.py`** – Converts raw ISMRMRD data to BART format, performs ESPIRiT coil sensitivity estimation and compressed sensing reconstruction per venc and temporal phase (with optional joint reconstruction and GPU acceleration), computes phase-difference velocities, and saves the result in [ORMIR-MIDS](https://github.com/ORMIR-MIDS/ORMIR_MIDS) NIfTI format.
  > **Note:** If converting from Siemens XA files, use this fork of the converter: [fsantini/siemens_to_ismrmrd](https://github.com/fsantini/siemens_to_ismrmrd)
- **`geometry.py`** – Extracts geometry information from ISMRMRD data to compute the affine matrix required for NIfTI export.
- **`calc_strain.py`** – Computes 3D displacement fields from velocity data and calculates the full strain tensor via Savitzky-Golay spatial derivatives, outputting principal strain eigenvalues and eigenvectors in [ORMIR-MIDS](https://github.com/ORMIR-MIDS/ORMIR_MIDS) NIfTI format.

## Installation

```bash
pip install -r requirements.txt
```

For BART toolbox installation, follow the instructions at the [BART website](https://mrirecon.github.io/bart/).

## Notes

These scripts are intended for research purposes.
