# Pulseq-Open4DFlow

Open4DFlow is an open-source, vendor-agnostic 4D flow MRI framework built with [PyPulseq](https://github.com/imr-framework/pypulseq). It provides ready-to-run pulse sequences and a full analysis pipeline, from acquisition to velocity reconstruction to diplacement ans strain calculation.

## Repository Structure

- **`sequences/`** – Pulse sequence scripts that generate `.seq` files for direct execution.
- **`analysis/`** – Post-processing pipeline for reconstructing velocity,displacement ans strain fields from acquired 4D flow data.

## Notes

These sequences are intended for research and educational purposes. Parameters, timing, and gradients should be carefully verified before in vivo use and potentially adapted to you system.

***
