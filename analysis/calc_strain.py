#!/usr/bin/env python3
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import argparse
import json
import nibabel as nib
from pathlib import Path


def sgolay3d(z, window_size, order, derivative=None):
    """Apply Savitzky-Golay filter to 3D data."""
    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    n_terms = (order + 1) * (order + 2) * (order + 3) // 6

    if window_size ** 3 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    exps = [(i - j - k, j, k) for i in range(order + 1) for j in range(i + 1) for k in range(i - j + 1)]

    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)

    xx, yy, zz = np.meshgrid(ind, ind, ind, indexing='ij')
    dx = xx.reshape(-1)
    dy = yy.reshape(-1)
    dz = zz.reshape(-1)

    A = np.empty((window_size ** 3, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1]) * (dz ** exp[2])

    new_shape = (z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size, z.shape[2] + 2 * half_size)
    Z = np.zeros(new_shape)

    Z[half_size:-half_size, half_size:-half_size, half_size:-half_size] = z

    band = z[:, :, 0]
    Z[half_size:-half_size, half_size:-half_size, :half_size] = np.stack([band] * half_size, axis=-1) - np.abs(
        np.flip(z[:, :, 1:half_size + 1], axis=2) - band[:, :, np.newaxis])

    band = z[:, :, -1]
    Z[half_size:-half_size, half_size:-half_size, -half_size:] = np.stack([band] * half_size, axis=-1) + np.abs(
        np.flip(z[:, :, -half_size - 1:-1], axis=2) - band[:, :, np.newaxis])

    band = z[:, 0, :]
    Z[half_size:-half_size, :half_size, half_size:-half_size] = np.stack([band] * half_size, axis=1) - np.abs(
        np.flip(z[:, 1:half_size + 1, :], axis=1) - band[:, np.newaxis, :])

    band = z[:, -1, :]
    Z[half_size:-half_size, -half_size:, half_size:-half_size] = np.stack([band] * half_size, axis=1) + np.abs(
        np.flip(z[:, -half_size - 1:-1, :], axis=1) - band[:, np.newaxis, :])

    band = z[0, :, :]
    Z[:half_size, half_size:-half_size, half_size:-half_size] = np.stack([band] * half_size, axis=0) - np.abs(
        np.flip(z[1:half_size + 1, :, :], axis=0) - band[np.newaxis, :, :])

    band = z[-1, :, :]
    Z[-half_size:, half_size:-half_size, half_size:-half_size] = np.stack([band] * half_size, axis=0) + np.abs(
        np.flip(z[-half_size - 1:-1, :, :], axis=0) - band[np.newaxis, :, :])

    pinv_A = np.linalg.pinv(A)

    if derivative is None:
        m = pinv_A[0].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'x':
        c = pinv_A[1].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'y':
        r = pinv_A[2].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'z':
        z_deriv = pinv_A[3].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, -z_deriv, mode='valid')
    elif derivative == 'all':
        x_deriv = pinv_A[1].reshape((window_size, window_size, window_size))
        y_deriv = pinv_A[2].reshape((window_size, window_size, window_size))
        z_deriv = pinv_A[3].reshape((window_size, window_size, window_size))
        return (
            scipy.signal.fftconvolve(Z, -x_deriv, mode='valid'),
            scipy.signal.fftconvolve(Z, -y_deriv, mode='valid'),
            scipy.signal.fftconvolve(Z, -z_deriv, mode='valid')
        )


def calc_disp_3d(flow_x, flow_y, flow_z, info, mask):
    """Calculate 3D displacement from flow velocity data (m/s input).

    flow_x/y/z: (x, y, z, t) velocity arrays in m/s
    info: dict with CardiacNumberOfImages, SliceThickness (mm), RepetitionTime (ms),
          InPlaneResolution ([res_x_mm, res_y_mm])
    Returns: dispVx, dispVy, dispVz in mm, shape (x, y, z, t)
    """

    def calcInterpolatedIndex3D(x, y, z, ActInterpFacX, ActInterpFacY, ActInterpFacZ):
        interpX = round((x - 1) * ActInterpFacX + 1)
        interpY = round((y - 1) * ActInterpFacY + 1)
        interpZ = round((z - 1) * ActInterpFacZ + 1)

        if interpX < 0:
            interpX = 0
        elif interpX > (interpDispX.shape[0] - 1):
            interpX = interpDispX.shape[0] - 1

        if interpY < 0:
            interpY = 0
        elif interpY > (interpDispX.shape[1] - 1):
            interpY = interpDispX.shape[1] - 1

        if interpZ < 0:
            interpZ = 0
        elif interpZ > (interpDispX.shape[2] - 1):
            interpZ = interpDispX.shape[2] - 1

        return interpX, interpY, interpZ

    nPhases = int(info['CardiacNumberOfImages'])
    SliceThickness = info['SliceThickness']
    dt = float(info['RepetitionTime'])  # ms
    print(f"Repetition time: {dt} ms")

    dimx = flow_x.shape[0]
    dimy = flow_x.shape[1]
    dimz = flow_x.shape[2]

    dispVx = np.zeros((dimx, dimy, dimz, nPhases))
    dispVy = np.zeros((dimx, dimy, dimz, nPhases))
    dispVz = np.zeros((dimx, dimy, dimz, nPhases))

    flow_x_corrected = flow_x.copy()
    flow_y_corrected = flow_y.copy()
    flow_z_corrected = flow_z.copy()

    flow_x_corrected -= np.mean(flow_x_corrected, axis=-1, keepdims=True)
    flow_y_corrected -= np.mean(flow_y_corrected, axis=-1, keepdims=True)
    flow_z_corrected -= np.mean(flow_z_corrected, axis=-1, keepdims=True)

    # velocity [m/s] * dt [ms] = displacement [mm]  (1 m/s * 1 ms = 1 mm)
    conversion_factor = 0.5 * dt

    diffDispX = flow_x_corrected * conversion_factor / info['InPlaneResolution'][0]
    diffDispY = flow_y_corrected * conversion_factor / info['InPlaneResolution'][1]
    diffDispZ = flow_z_corrected * conversion_factor / SliceThickness

    print(f"Differential displacement shape: {diffDispX.shape}")

    xnew = np.linspace(0, dimx - 1, int(dimx * 1.25))
    ynew = np.linspace(0, dimy - 1, int(dimy * 1.25))
    znew = np.linspace(0, dimz - 1, int(dimz * 1.25))

    X, Y, Z = np.meshgrid(xnew, ynew, znew, indexing='ij')

    interpDispX = np.zeros((len(xnew), len(ynew), len(znew), nPhases))
    interpDispY = np.zeros((len(xnew), len(ynew), len(znew), nPhases))
    interpDispZ = np.zeros((len(xnew), len(ynew), len(znew), nPhases))

    print(f"Number of phases: {nPhases}")

    x = np.linspace(0, dimx - 1, dimx)
    y = np.linspace(0, dimy - 1, dimy)
    z = np.linspace(0, dimz - 1, dimz)
    for iph in range(nPhases):
        points = (x, y, z)
        interp_x_func = scipy.interpolate.RegularGridInterpolator(points, diffDispX[:, :, :, iph], bounds_error=False)
        interp_y_func = scipy.interpolate.RegularGridInterpolator(points, diffDispY[:, :, :, iph], bounds_error=False)
        interp_z_func = scipy.interpolate.RegularGridInterpolator(points, diffDispZ[:, :, :, iph], bounds_error=False)

        pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        interpDispX[:, :, :, iph] = interp_x_func(pts).reshape(X.shape)
        interpDispY[:, :, :, iph] = interp_y_func(pts).reshape(Y.shape)
        interpDispZ[:, :, :, iph] = interp_z_func(pts).reshape(Z.shape)

    ActInterpFacX = len(xnew) / len(x)
    ActInterpFacY = len(ynew) / len(y)
    ActInterpFacZ = len(znew) / len(z)

    print(f"ActInterpFacX: {ActInterpFacX}, ActInterpFacY: {ActInterpFacY}, ActInterpFacZ: {ActInterpFacZ}")

    for iph in range(1, nPhases):
        for ix in range(dimx):
            for iy in range(dimy):
                for iz in range(dimz):
                    if mask[ix, iy, iz]:
                        new_xn = ix + dispVx[ix, iy, iz, iph]
                        new_yn = iy + dispVy[ix, iy, iz, iph]
                        new_zn = iz + dispVz[ix, iy, iz, iph]

                        new_xnm1 = ix - dispVx[ix, iy, iz, iph - 1]
                        new_ynm1 = iy - dispVy[ix, iy, iz, iph - 1]
                        new_znm1 = iz - dispVz[ix, iy, iz, iph - 1]

                        inew_xn, inew_yn, inew_zn = calcInterpolatedIndex3D(
                            new_xn, new_yn, new_zn,
                            ActInterpFacX, ActInterpFacY, ActInterpFacZ)

                        inew_xnm1, inew_ynm1, inew_znm1 = calcInterpolatedIndex3D(
                            new_xnm1, new_ynm1, new_znm1,
                            ActInterpFacX, ActInterpFacY, ActInterpFacZ)

                        forward_velocity_x = interpDispX[inew_xn, inew_yn, inew_zn, iph]
                        forward_velocity_y = interpDispY[inew_xn, inew_yn, inew_zn, iph]
                        forward_velocity_z = interpDispZ[inew_xn, inew_yn, inew_zn, iph]

                        backward_velocity_x = interpDispX[inew_xnm1, inew_ynm1, inew_znm1, iph - 1]
                        backward_velocity_y = interpDispY[inew_xnm1, inew_ynm1, inew_znm1, iph - 1]
                        backward_velocity_z = interpDispZ[inew_xnm1, inew_ynm1, inew_znm1, iph - 1]

                        delta_disp_x = backward_velocity_x + forward_velocity_x
                        delta_disp_y = backward_velocity_y + forward_velocity_y
                        delta_disp_z = backward_velocity_z + forward_velocity_z

                        dispVx[ix, iy, iz, iph] = dispVx[ix, iy, iz, iph - 1] + delta_disp_x
                        dispVy[ix, iy, iz, iph] = dispVy[ix, iy, iz, iph - 1] + delta_disp_y
                        dispVz[ix, iy, iz, iph] = dispVz[ix, iy, iz, iph - 1] + delta_disp_z

    print(f"Displacement shapes: {dispVx.shape}, {dispVy.shape}, {dispVz.shape}")

    dispVxi_mm = info['InPlaneResolution'][0] * dispVx
    dispVyi_mm = info['InPlaneResolution'][1] * dispVy
    dispVzi_mm = SliceThickness * dispVz

    return dispVxi_mm.astype(np.float32), dispVyi_mm.astype(np.float32), dispVzi_mm.astype(np.float32)


def calc_strain_3d(dispX, dispY, dispZ, mask):
    """Calculate 3D strain tensor eigenvalues and eigenvectors.

    Returns:
        Eig_v:    (x, y, z, 3, t) eigenvalues ordered per ORMIR-MIDS spec
        Eig_vecs: (x, y, z, 3, 3, t) eigenvectors [ev_order, vector_component, t]
    """

    def strain3D(dispX_3D, dispY_3D, dispZ_3D):
        dimx, dimy, dimz = dispX_3D.shape
        E = np.zeros((dimx, dimy, dimz, 3, 3))

        Uxx = sgolay3d(dispX_3D, window_size=13, order=4, derivative='x')
        Uxy = sgolay3d(dispX_3D, window_size=13, order=4, derivative='y')
        Uxz = sgolay3d(dispX_3D, window_size=13, order=4, derivative='z')

        Uyx = sgolay3d(dispY_3D, window_size=13, order=4, derivative='x')
        Uyy = sgolay3d(dispY_3D, window_size=13, order=4, derivative='y')
        Uyz = sgolay3d(dispY_3D, window_size=13, order=4, derivative='z')

        Uzx = sgolay3d(dispZ_3D, window_size=13, order=4, derivative='x')
        Uzy = sgolay3d(dispZ_3D, window_size=13, order=4, derivative='y')
        Uzz = sgolay3d(dispZ_3D, window_size=13, order=4, derivative='z')

        for ix in range(dimx):
            for iy in range(dimy):
                for iz in range(dimz):
                    Ugrad = np.array([
                        [Uxx[ix, iy, iz], Uxy[ix, iy, iz], Uxz[ix, iy, iz]],
                        [Uyx[ix, iy, iz], Uyy[ix, iy, iz], Uyz[ix, iy, iz]],
                        [Uzx[ix, iy, iz], Uzy[ix, iy, iz], Uzz[ix, iy, iz]]
                    ])
                    Finv = np.eye(3) - Ugrad
                    e = 0.5 * (np.eye(3) - Finv @ Finv.T)
                    E[ix, iy, iz, :, :] = e

        return E

    dimx, dimy, dimz = dispX.shape[:3]
    n_phases = dispX.shape[3] if dispX.ndim > 3 else 1

    # Internal storage: (x, y, z, ev_order, t) and (x, y, z, ev_order, vec_component, t)
    Eig_v = np.zeros((dimx, dimy, dimz, 3, n_phases))
    Eig_vecs = np.zeros((dimx, dimy, dimz, 3, 3, n_phases))

    for iph in range(n_phases):
        dispX_phase = dispX[:, :, :, iph] if n_phases > 1 else dispX
        dispY_phase = dispY[:, :, :, iph] if n_phases > 1 else dispY
        dispZ_phase = dispZ[:, :, :, iph] if n_phases > 1 else dispZ

        strain_tensor = strain3D(dispX_phase, dispY_phase, dispZ_phase)

        skipped = 0
        for ix in range(dimx):
            for iy in range(dimy):
                for iz in range(dimz):
                    if mask[ix, iy, iz]:
                        # eigh: eigenvalues in ascending order, eigenvectors as columns
                        eigenvalues, eigenvectors = np.linalg.eigh(strain_tensor[ix, iy, iz])

                        # ORMIR-MIDS order: largest positive, most negative (largest abs negative), remaining
                        idx_pos = int(np.argmax(eigenvalues))
                        idx_neg = int(np.argmin(eigenvalues))
                        if idx_pos == idx_neg:
                            skipped += 1
                            continue
                        idx_rem = 3 - idx_pos - idx_neg
                        order = [idx_pos, idx_neg, idx_rem]

                        Eig_v[ix, iy, iz, :, iph] = eigenvalues[order]
                        # eigenvectors[:, i] is i-th vector; store as (ev_order, vec_component)
                        Eig_vecs[ix, iy, iz, :, :, iph] = eigenvectors[:, order].T
    print("Total skipped voxels:", skipped)
    return Eig_v, Eig_vecs


def sum_strain_3d(Eig_v, mask, n_phases):
    """Summarize strain eigenvalues by computing median within mask per phase.

    Eig_v: (x, y, z, 3, t) internal shape
    Returns: (e1_Line, e1_max, e2_Line, e2_max, e3_Line, e3_max)
    """
    e1_Line = np.zeros(n_phases)
    e2_Line = np.zeros(n_phases)
    e3_Line = np.zeros(n_phases)

    for i_ph in range(n_phases):
        E1 = Eig_v[:, :, :, 0, i_ph]
        E2 = Eig_v[:, :, :, 1, i_ph]
        E3 = Eig_v[:, :, :, 2, i_ph]

        masked_E1 = E1[mask]
        masked_E2 = E2[mask]
        masked_E3 = E3[mask]

        if len(masked_E1) > 0:
            e1_Line[i_ph] = np.nanmedian(masked_E1[masked_E1 != 0])
        if len(masked_E2) > 0:
            e2_Line[i_ph] = np.nanmedian(masked_E2[masked_E2 != 0])
        if len(masked_E3) > 0:
            e3_Line[i_ph] = np.nanmedian(masked_E3[masked_E3 != 0])

    e1_max = np.round(np.nanmax(e1_Line), 3)
    e2_max = np.round(np.nanmax(np.abs(e2_Line)), 3)
    e3_max = np.round(np.nanmax(e3_Line), 3)

    return e1_Line, e1_max, e2_Line, e2_max, e3_Line, e3_max


def load_ormir_mids_velocity(input_path: Path):
    """Load velocity NIfTI from ORMIR-MIDS format.

    input_path: path to *_vel.nii.gz file, or an ORMIR-MIDS subject folder
                (in which case mr-quant/*_vel.nii.gz is used)

    Returns: (velocities, affine, header, sidecar, vel_path)
        velocities: (x, y, z, t, 3) float array in m/s
        affine:     4x4 affine matrix
        header:     NIfTI header object
        sidecar:    dict loaded from the _vel.json sidecar
        vel_path:   Path to the resolved *_vel.nii.gz file
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        mr_quant = input_path / "mr-quant"
        vel_files = sorted(mr_quant.glob("*_vel.nii.gz"))
        if not vel_files:
            raise FileNotFoundError(f"No *_vel.nii.gz files found in {mr_quant}")
        vel_path = vel_files[0]
        if len(vel_files) > 1:
            print(f"  WARNING: Multiple velocity files found, using {vel_path.name}")
    else:
        vel_path = input_path

    json_path = vel_path.with_name(vel_path.name.replace("_vel.nii.gz", "_vel.json"))
    if not json_path.exists():
        raise FileNotFoundError(f"Sidecar JSON not found: {json_path}")

    img = nib.load(str(vel_path))
    velocities = img.get_fdata()
    affine = img.affine
    header = img.header

    with open(json_path) as f:
        sidecar = json.load(f)

    print(f"Loaded velocity: {vel_path.name}  shape={velocities.shape}")
    return velocities, affine, header, sidecar, vel_path


def save_ormir_mids_strain(vel_path: Path, stem: str,
                           eigenvalues_mids: np.ndarray, eigenvectors_mids: np.ndarray,
                           affine: np.ndarray, trigger_times: list):
    """Save strain data in ORMIR-MIDS format to the same mr-quant folder as the velocity file.

    eigenvalues_mids:  (x, y, z, t, ev_order)
    eigenvectors_mids: (x, y, z, t, ev_order, vector_component)
    """
    bids_dir = vel_path.parent  # already the mr-quant folder
    bids_dir.mkdir(exist_ok=True)

    nib.save(
        nib.Nifti1Image(eigenvalues_mids.astype(np.float32), affine),
        str(bids_dir / f"{stem}_strain.nii.gz")
    )
    with open(bids_dir / f"{stem}_strain.json", 'w') as f:
        json.dump({
            "FourthDimension": "TriggerTime",
            "TriggerTime": trigger_times,
            "FifthDimension": "EigenOrder",
        }, f, indent=2)

    nib.save(
        nib.Nifti1Image(eigenvectors_mids.astype(np.float32), affine),
        str(bids_dir / f"{stem}_strain-vec.nii.gz")
    )
    with open(bids_dir / f"{stem}_strain-vec.json", 'w') as f:
        json.dump({
            "FourthDimension": "TriggerTime",
            "TriggerTime": trigger_times,
            "FifthDimension": "EigenOrder",
            "SixthDimension": "VectorComponent",
        }, f, indent=2)

    print(f"  {stem}_strain.nii.gz    {eigenvalues_mids.shape}")
    print(f"  {stem}_strain.json")
    print(f"  {stem}_strain-vec.nii.gz {eigenvectors_mids.shape}")
    print(f"  {stem}_strain-vec.json")


def calc_strain_pipeline(velocities_ms: np.ndarray, affine: np.ndarray,
                         header, sidecar: dict):
    """Run the full strain pipeline on ORMIR-MIDS velocity data.

    velocities_ms: (x, y, z, t, 3) in m/s
    header:        NIfTI header (used to extract voxel size)
    sidecar:       _vel.json dict (used to extract TriggerTime)

    Returns:
        eig_vals_mids:  (x, y, z, t, ev_order)   ORMIR-MIDS shape
        eig_vecs_mids:  (x, y, z, t, ev_order, 3) ORMIR-MIDS shape
        mask:           (x, y, z) boolean
    """
    n_phases = velocities_ms.shape[3]

    # Voxel sizes from NIfTI header (mm)
    pixdim = header.get_zooms()
    in_plane_res = [float(pixdim[0]), float(pixdim[1])]
    slice_thickness = float(pixdim[2])
    print(f"Voxel size (mm): {in_plane_res[0]:.3f} x {in_plane_res[1]:.3f} x {slice_thickness:.3f}")
    if abs(in_plane_res[0] - 1.0) < 1e-6 and abs(in_plane_res[1] - 1.0) < 1e-6 and abs(slice_thickness - 1.0) < 1e-6:
        raise ValueError(
            "Voxel size appears to be 1x1x1 mm (identity); check NIfTI header pixdim. "
            "Displacement calculation requires accurate voxel dimensions."
        )

    # TR from consecutive TriggerTime values
    trigger_times = sidecar.get("TriggerTime", [])
    if len(trigger_times) >= 2:
        tr_ms = float(trigger_times[1]) - float(trigger_times[0])
    else:
        tr_ms = 7.0
        print(f"  WARNING: TriggerTime not found in sidecar, defaulting TR to {tr_ms} ms")

    info = {
        'CardiacNumberOfImages': n_phases,
        'SliceThickness': slice_thickness,
        'RepetitionTime': tr_ms,
        'InPlaneResolution': in_plane_res,
    }

    # Recover mask from zero-ed velocity voxels (reconstruct_and_process zeros outside mask)
    mask = np.any(velocities_ms != 0, axis=(3, 4))
    mask_n_voxel = np.sum(mask)
    total_voxel = np.prod(velocities_ms.shape[:3])
    print(f"Mask: {np.sum(mask)} voxels ({float(mask_n_voxel)/total_voxel*100:.2f}%)")

    flow_x = velocities_ms[..., 0]
    flow_y = velocities_ms[..., 1]
    flow_z = velocities_ms[..., 2]

    print("Calculating displacements...")
    dispVxi, dispVyi, dispVzi = calc_disp_3d(flow_x, flow_y, flow_z, info, mask)

    print("Calculating strain...")
    Eig_v, Eig_vecs = calc_strain_3d(dispVxi, dispVyi, dispVzi, mask)

    # Reorder to ORMIR-MIDS axis convention
    eig_vals_mids = Eig_v.transpose(0, 1, 2, 4, 3)          # (x,y,z,3,t) -> (x,y,z,t,3)
    eig_vecs_mids = Eig_vecs.transpose(0, 1, 2, 5, 3, 4)    # (x,y,z,3,3,t) -> (x,y,z,t,3,3)

    return eig_vals_mids, eig_vecs_mids, mask


def main():
    parser = argparse.ArgumentParser(description='Calculate 3D strain from ORMIR-MIDS velocity data.')
    parser.add_argument('input', type=str,
                        help='Path to *_vel.nii.gz file, or ORMIR-MIDS subject folder '
                             '(mr-quant/*_vel.nii.gz is used in that case)')
    parser.add_argument('--output-plot', help='Save strain plot to this file (.png)')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots')
    args = parser.parse_args()

    velocities_ms, affine, header, sidecar, vel_path = load_ormir_mids_velocity(Path(args.input))

    stem = vel_path.name.replace("_vel.nii.gz", "")

    eig_vals_mids, eig_vecs_mids, mask = calc_strain_pipeline(velocities_ms, affine, header, sidecar)

    trigger_times = sidecar.get("TriggerTime", list(range(eig_vals_mids.shape[3])))
    save_ormir_mids_strain(vel_path, stem, eig_vals_mids, eig_vecs_mids, affine, trigger_times)

    n_phases = eig_vals_mids.shape[3]
    # Internal shape (x,y,z,3,t) needed by sum_strain_3d
    Eig_v_internal = eig_vals_mids.transpose(0, 1, 2, 4, 3)
    e1_Line, e1_max, e2_Line, e2_max, e3_Line, e3_max = sum_strain_3d(Eig_v_internal, mask, n_phases)

    print(f"Largest positive strain max: {e1_max}")
    print(f"Most negative strain max:    {e2_max}")
    print(f"Remaining strain max:        {e3_max}")

    if not args.no_display or args.output_plot:
        plt.figure(figsize=(10, 6))
        for data, label, color in zip(
            [e1_Line, e2_Line, e3_Line],
            ['e1 (largest positive)', 'e2 (most negative)', 'e3 (remaining)'],
            ['blue', 'red', 'green']
        ):
            plt.plot(data, label=label, color=color)

        plt.xlabel('Phase')
        plt.ylabel('Strain')
        plt.title('Principal Strains')
        plt.legend()
        plt.grid(True)

        if args.output_plot:
            plt.savefig(args.output_plot, dpi=300)

        if not args.no_display:
            plt.show()


if __name__ == "__main__":
    main()