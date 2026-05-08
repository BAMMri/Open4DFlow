#!/usr/bin/env python3
"""
run_pipeline.py — 4D Flow MRI end-to-end pipeline
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths to your pipeline scripts
# ---------------------------------------------------------------------------
RECON_SCRIPT = Path("/home/marta/PycharmProjects/InhomogeneousMT/reconstruct_and_process.py")
STRAIN_SCRIPT = Path("/home/marta/PycharmProjects/InhomogeneousMT/calc_strain.py")


# =============================================================================
# STEP 1 — Geometry extraction → metadata JSON
# =============================================================================

def extract_geometry_and_build_json(mrd_path: Path, output_dir: Path, config: dict) -> Path:
    print("\n[1/5] Extracting geometry and merging with config...")

    affine_list = None
    pixel_size = None
    fov = None
    table_pos = None
    patient_position = "unknown"

    try:
        import h5py
        import ismrmrd
        from geometry import Geometry

        with h5py.File(mrd_path, 'r') as f:
            xml_bytes = np.array(f['dataset']['xml'][0])

        dset = ismrmrd.Dataset(str(mrd_path))
        geo = Geometry()
        geo.from_ismrmrd(dset, xml_bytes)

        affine_list = geo.get_dcm().tolist()
        pixel_size = [
            geo.fov[0] / geo.matrix_size[0],
            geo.fov[1] / geo.matrix_size[1],
            geo.fov[2] / geo.matrix_size[2],
        ]
        fov = geo.fov
        table_pos = geo.table_position
        patient_position = geo.patient_position
    except Exception as e:
        print(f"  WARNING: Could not extract geometry ({e}).")

    try:
        # Start with config values and update with extracted geometry
        metadata = config.copy()
        metadata.update({
            "RepetitionTime": metadata.get("TR", 0.007) * 1000,
            "PatientPosition": patient_position,
            "AffineMatrix_4x4": affine_list,
            "PixelSize_mm": pixel_size,
            "FieldOfView_mm": fov,
            "TablePosition_mm": table_pos,
            "SourceFile": mrd_path.name,
            "ProcessingDate": datetime.now().isoformat(),
            "BART_venc_cmps": metadata.get("venc", 150),
            "BART_ecalib": metadata.get("ecalib_r", "20")  # Default if missing
        })

        final_json_path = output_dir / f"{mrd_path.stem}_reconstruction_metadata.json"
        with open(final_json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Geometry merged. New metadata: {final_json_path.name}")
        return final_json_path

    except Exception as e:
        print(f"  CRITICAL ERROR: Could not process metadata: {e}")
        raise


# =============================================================================
# STEP 2 — MRD → BART conversion
# =============================================================================

def convert_mrd_to_bart(mrd_path: Path, output_dir: Path) -> Path:
    print("\n[2/5] Converting .mrd → BART format...")
    import copy
    import ismrmrd
    from tqdm import tqdm

    output_stem = output_dir / mrd_path.stem
    cfl_path = Path(str(output_stem) + '.cfl')
    hdr_path = Path(str(output_stem) + '.hdr')

    if cfl_path.exists() and hdr_path.exists():
        print(f"  BART files exist, skipping conversion.")
        return output_dir

    dataset = ismrmrd.Dataset(str(mrd_path))
    acquisition_list = []

    # Simple shape discovery
    acq0 = dataset.read_acquisition(0)
    channels, matrix_x = acq0.data.shape
    m_y = m_z = c = ph = r = s = seg = sl = avg = 1

    for idx in tqdm(range(dataset.number_of_acquisitions())):
        acq = dataset.read_acquisition(idx)
        idx_c = acq.getHead().idx
        m_y, m_z = max(m_y, idx_c.kspace_encode_step_1 + 1), max(m_z, idx_c.kspace_encode_step_2 + 1)
        c, ph = max(c, idx_c.contrast + 1), max(ph, idx_c.phase + 1)
        r, s = max(r, idx_c.repetition + 1), max(s, idx_c.set + 1)
        seg, sl, avg = max(seg, idx_c.segment + 1), max(sl, idx_c.slice + 1), max(avg, idx_c.average + 1)
        acquisition_list.append((copy.deepcopy(idx_c), np.copy(acq.data.astype('<c8'))))

    out_shape = (matrix_x, m_y, m_z, channels, 1, c, ph, 1, 1, 1, r, s, seg, sl, avg, 1)
    output_mmap = np.memmap(str(cfl_path), dtype='<c8', mode='w+', shape=out_shape, order='F')

    for counters, data in tqdm(acquisition_list):
        output_mmap[:, counters.kspace_encode_step_1, counters.kspace_encode_step_2, :, 0,
        counters.contrast, counters.phase, 0, 0, 0, counters.repetition,
        counters.set, counters.segment, counters.slice, counters.average, 0] = data.T

    with open(str(hdr_path), 'w') as f:
        f.write('# Dimensions\n' + ' '.join(map(str, out_shape)) + '\n')

    return output_dir


# =============================================================================
# STEP 3 — Reconstruction
# =============================================================================

def run_reconstruction(bart_dir: Path, config: dict) -> Path:
    print("\n[3/5] Running 4D flow reconstruction...")
    if not RECON_SCRIPT.exists():
        raise FileNotFoundError(f"Missing: {RECON_SCRIPT}")

    ecalib = config.get("fullysampled_center", [13, 10])
    if isinstance(ecalib, str):
        ecalib = json.loads(ecalib)
    ecalib_r = ":".join(map(str, ecalib)) if isinstance(ecalib, list) else str(ecalib)

    venc = float(config.get("venc", 1.5)) * 100  # m/s -> cm/s

    cmd = [
        sys.executable, str(RECON_SCRIPT), str(bart_dir),
        "--venc",     str(venc),
        "--ecalib-r", ecalib_r,
        "--cc",       str(config.get("compressed_coils", 8))
    ]
    if not config.get("use_gpu", True):
        cmd.append("--no-gpu")
    if config.get("joint_recon", False):
        cmd.append("--joint")

    subprocess.run(cmd, check=True)
    return bart_dir


# =============================================================================
# STEP 4 & 5 — Strain and NIfTI
# =============================================================================

def run_strain(data_dir: Path, json_path: Path):
    print("\n[4/5] Calculating displacement and strain...")
    eig_out, plot_out = data_dir / "Eig_v_output.npy", data_dir / "strain_plot.png"
    cmd = [
        sys.executable, str(STRAIN_SCRIPT),
        "--data-path", str(data_dir), "--config", str(json_path),
        "--output-eig", str(eig_out), "--output-plot", str(plot_out), "--no-display"
    ]
    subprocess.run(cmd, check=True)


def save_niftis(data_dir: Path, json_path: Path):
    print("\n[5/5] Saving NIfTI files...")
    import nibabel as nib
    import glob
    with open(json_path) as f:
        meta = json.load(f)
    affine = np.array(meta["AffineMatrix_4x4"])

    # Save velocities
    for f_path in glob.glob(str(data_dir / "*_processed_data.npy")):
        proc = np.load(f_path, allow_pickle=True).item()
        nib.save(nib.Nifti1Image(proc['velocities'].astype(np.float32), affine), str(data_dir / "velocities.nii.gz"))
        if 'mask' in proc:
            nib.save(nib.Nifti1Image(proc['mask'].astype(np.float32), affine), str(data_dir / "mask.nii.gz"))

    # Save eigenvalues
    eig_path = data_dir / "Eig_v_output.npy"
    if eig_path.exists():
        eig = np.moveaxis(np.load(str(eig_path)), 3, 4)
        nib.save(nib.Nifti1Image(eig.astype(np.float32), affine), str(data_dir / "eigenvalues.nii.gz"))


def save_muscle_bids(data_dir: Path, json_path: Path):
    print("\n[4c/5] Saving muscle-BIDS format...")
    import nibabel as nib
    import glob

    with open(json_path) as f:
        meta = json.load(f)
    affine_matrix = np.array(meta["AffineMatrix_4x4"])
    venc = meta.get("BART_venc_cmps", 150)  # cm/s
    venc_ms = venc / 100  # convert to m/s

    # Create BIDS directory
    bids_dir = data_dir / "mr-quant"
    bids_dir.mkdir(exist_ok=True)
    stem = data_dir.name  # use output dir name as subject stem

    # --- Velocity NIfTI: shape must be (x, y, z, t, direction) ---
    proc_files = glob.glob(str(data_dir / "*_processed_data.npy"))
    if proc_files:
        proc = np.load(proc_files[0], allow_pickle=True).item()
        velocities = proc['velocities']  # (100, 67, 54, 27, 3) - already correct shape

        nib.save(
            nib.Nifti1Image(velocities.astype(np.float32), affine_matrix),
            str(bids_dir / f"{stem}_vel.nii.gz")
        )

        n_timepoints = velocities.shape[3]
        trigger_times = list(np.arange(n_timepoints) * meta.get("RepetitionTime", 7.0))

        vel_sidecar = {
            "FourthDimension": "TriggerTime",
            "TriggerTime": trigger_times,
            "FifthDimension": "VelocityEncodingDirection",
            "VelocityEncodingDirection": [
                [1, 0, 0],  # x / read
                [0, 1, 0],  # y / phase
                [0, 0, 1],  # z / slice
            ],
            "Venc": [venc_ms, venc_ms, venc_ms],
            "Units": "m/s",
            "AffineMatrix_4x4": meta["AffineMatrix_4x4"],
            "PixelSize_mm": meta.get("PixelSize_mm"),
            "FieldOfView_mm": meta.get("FieldOfView_mm"),
            "PatientPosition": meta.get("PatientPosition"),
        }

        with open(bids_dir / f"{stem}_vel.json", 'w') as f:
            json.dump(vel_sidecar, f, indent=2)

        print(f"  ✓ {stem}_vel.nii.gz {velocities.shape}")
        print(f"  ✓ {stem}_vel.json")

    # --- Eigenvalues ---
    eig_path = data_dir / "Eig_v_output.npy"
    if eig_path.exists():
        eig = np.load(str(eig_path))
        eig = np.moveaxis(eig, 3, 4)  # (x,y,z,3,27) -> (x,y,z,27,3)

        nib.save(
            nib.Nifti1Image(eig.astype(np.float32), affine_matrix),
            str(bids_dir / f"{stem}_strain.nii.gz")
        )

        eig_sidecar = {
            "FourthDimension": "TriggerTime",
            "TriggerTime": trigger_times,
            "FifthDimension": "StrainEigenvalues",
            "Units": "strain",
            "AffineMatrix_4x4": meta["AffineMatrix_4x4"],
            "PatientPosition": meta.get("PatientPosition"),
        }

        with open(bids_dir / f"{stem}_strain.json", 'w') as f:
            json.dump(eig_sidecar, f, indent=2)

        print(f"  ✓ {stem}_strain.nii.gz {eig.shape}")
        print(f"  ✓ {stem}_strain.json")



# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="4D Flow MRI pipeline")
    parser.add_argument("--mrd", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True, help="Path to derived_params.json")
    parser.add_argument("--skip-recon", action="store_true")
    parser.add_argument("--skip-strain", action="store_true")
    parser.add_argument("--save-bids", action="store_true")
    args = parser.parse_args()

    mrd_path, output_dir = Path(args.mrd).resolve(), Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Step 1: geometry + merge
    json_path = extract_geometry_and_build_json(mrd_path, output_dir, config)

    # Step 2-3: Recon
    if not args.skip_recon:
        bart_dir = convert_mrd_to_bart(mrd_path, output_dir)
        run_reconstruction(bart_dir, config)

    # Step 4-5: Analysis
    if not args.skip_strain:
        run_strain(output_dir, json_path)
        save_niftis(output_dir, json_path)

    if args.save_bids:
        save_muscle_bids(output_dir, json_path)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()