# Open4DFlow

Open4DFlow is an open-source, vendor-agnostic 4D flow MRI framework built with [PyPulseq](https://github.com/imr-framework/pypulseq). It provides ready-to-run pulse sequences and a full analysis pipeline, from acquisition to velocity reconstruction to diplacement ans strain calculation.

## Repository Structure

- **`sequences/`** – Pulse sequence scripts that generate `.seq` files for direct execution.
- **`analysis/`** – Post-processing pipeline for reconstructing velocity,displacement ans strain fields from acquired 4D flow data.

## Features

- **Compressed sensing acceleration** – The undersampled sequences use variable-density k-space undersampling with iterative reconstruction via [BART](https://mrirecon.github.io/bart/), enabling significantly faster acquisitions.
- **Vendor-agnostic gradient handling** – A gradient probing sequence identifies the physical direction of each gradient axis at the used scanner, ensuring correct velocity direction interpretation across scanner vendors.
- **Standardised file formats** – The analysis pipeline outputs data in [ORMIR-MIDS](https://github.com/ORMIR-MIDS/ORMIR_MIDS) format (NIfTI + JSON sidecar), ensuring compatibility with other muscle and cardiovascular MRI tools.
- **Open reconstruction** – Image reconstruction is performed with the [BART toolbox](https://mrirecon.github.io/bart/), with optional GPU acceleration.

## Installation and Usage

See the README files in the corresponding subfolders for more information on how to install and run each component of the pipeline.

## Notes

These sequences are intended for research and educational purposes. Parameters, timing, and gradients should be carefully verified before in vivo use and potentially adapted to you system.

***
