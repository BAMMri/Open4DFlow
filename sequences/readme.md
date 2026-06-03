# Open4DFlow — MRI Pulse Sequences

This folder contains open-source 4D flow MRI pulse sequences. 

---

## Sequences

### `gradient_probing.py` — Gradient mapping 
The gradient probing sequence determines the physical polarity of each gradient axis per vendor/orientation, so that velocity directions in the 4D flow data are interpreted correctly during reconstruction.

### `4dflow.py` — Standard 4D Flow Sequence
The main 4D flow sequence. This is the **fully sampled** version, presented here Maggioni MB, Räuber SM, Santini F. Vendor-agnostic, open-source 4D flow. In: Proceedings of the International Society for Magnetic Resonance in Medicine. 2025;33:Abstract 3208.

### `undersampling_forearm.py`, `undersampling_leg.py` and `undersampling_arteries.py`
Variants of the undersampled sequence adapted for various applications. These sequences were specifically used for acquisitions in the **forearm** and **leg**, with field-of-view, resolution, and encoding velocity parameters tuned accordingly, and for  **neurovascular imaging**. 

---

## Dependencies

Install all Python requirements with:

```bash
pip install -r requirements.txt
```

### GrOpt (Gradient Optimization)

The sequences rely on **GrOpt** for time-optimal gradient waveform design.

Install GrOpt by building the Cython wrappers from the `python/` directory of the GrOpt repository:

```bash
git clone https://github.com/mloecher/gropt
cd gropt/python
python setup.py install
```

---

## Usage

Each sequence script can be run directly from the terminal:

```bash
python 4Dflow.py
```

This generates a `.seq` file (Pulseq format) that can be transferred to the scanner. For instructions on running Pulseq sequences on a scanner, see the [Pulseq tutorials](https://github.com/pulseq/tutorials).

---

## Citation

If you use these sequences in your work, please cite the relevant BAMMri publications

## License

See [LICENSE](../LICENSE) in the root of the repository.
