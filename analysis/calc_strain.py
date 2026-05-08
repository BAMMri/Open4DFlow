#!/usr/bin/env python3
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import argparse
import os.path
import glob
import json


def sgolay3d(z, window_size, order, derivative=None):
    """
    Apply Savitzky-Golay filter to 3D data.
    """
    # Check window size is odd
    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    # Number of terms in the polynomial expression for 3D
    n_terms = (order + 1) * (order + 2) * (order + 3) // 6

    if window_size ** 3 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # Exponents of the polynomial (x^a * y^b * z^c)
    # Generate exponents for terms up to specified order
    exps = [(i - j - k, j, k) for i in range(order + 1) for j in range(i + 1) for k in range(i - j + 1)]

    # Coordinates of points in the cube window
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)

    # Create coordinate grids correctly
    xx, yy, zz = np.meshgrid(ind, ind, ind, indexing='ij')
    dx = xx.reshape(-1)  # Flatten to 1D
    dy = yy.reshape(-1)  # Flatten to 1D
    dz = zz.reshape(-1)  # Flatten to 1D

    # Build matrix of system of equations
    A = np.empty((window_size ** 3, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1]) * (dz ** exp[2])

    # Pad input array with appropriate values at the six faces and corners
    new_shape = (z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size, z.shape[2] + 2 * half_size)
    Z = np.zeros(new_shape)

    # Insert the original array in the center
    Z[half_size:-half_size, half_size:-half_size, half_size:-half_size] = z

    # Pad the six faces (front, back, top, bottom, left, right)
    # Front face (z=0)
    band = z[:, :, 0]
    Z[half_size:-half_size, half_size:-half_size, :half_size] = np.stack([band] * half_size, axis=-1) - np.abs(
        np.flip(z[:, :, 1:half_size + 1], axis=2) - band[:, :, np.newaxis])

    # Back face (z=max)
    band = z[:, :, -1]
    Z[half_size:-half_size, half_size:-half_size, -half_size:] = np.stack([band] * half_size, axis=-1) + np.abs(
        np.flip(z[:, :, -half_size - 1:-1], axis=2) - band[:, :, np.newaxis])

    # Top face (y=0)
    band = z[:, 0, :]
    Z[half_size:-half_size, :half_size, half_size:-half_size] = np.stack([band] * half_size, axis=1) - np.abs(
        np.flip(z[:, 1:half_size + 1, :], axis=1) - band[:, np.newaxis, :])

    # Bottom face (y=max)
    band = z[:, -1, :]
    Z[half_size:-half_size, -half_size:, half_size:-half_size] = np.stack([band] * half_size, axis=1) + np.abs(
        np.flip(z[:, -half_size - 1:-1, :], axis=1) - band[:, np.newaxis, :])

    # Left face (x=0)
    band = z[0, :, :]
    Z[:half_size, half_size:-half_size, half_size:-half_size] = np.stack([band] * half_size, axis=0) - np.abs(
        np.flip(z[1:half_size + 1, :, :], axis=0) - band[np.newaxis, :, :])

    # Right face (x=max)
    band = z[-1, :, :]
    Z[-half_size:, half_size:-half_size, half_size:-half_size] = np.stack([band] * half_size, axis=0) + np.abs(
        np.flip(z[-half_size - 1:-1, :, :], axis=0) - band[np.newaxis, :, :])

    # Solve system and convolve
    pinv_A = np.linalg.pinv(A)

    if derivative is None:
        # For smoothing only
        m = pinv_A[0].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'x':
        # x derivative
        c = pinv_A[1].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'y':
        # y derivative
        r = pinv_A[2].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'z':
        # z derivative
        z_deriv = pinv_A[3].reshape((window_size, window_size, window_size))
        return scipy.signal.fftconvolve(Z, -z_deriv, mode='valid')
    elif derivative == 'all':
        # All derivatives
        x_deriv = pinv_A[1].reshape((window_size, window_size, window_size))
        y_deriv = pinv_A[2].reshape((window_size, window_size, window_size))
        z_deriv = pinv_A[3].reshape((window_size, window_size, window_size))
        return (
            scipy.signal.fftconvolve(Z, -x_deriv, mode='valid'),
            scipy.signal.fftconvolve(Z, -y_deriv, mode='valid'),
            scipy.signal.fftconvolve(Z, -z_deriv, mode='valid')
        )


def calc_disp_3d(flow_x, flow_y, flow_z, info, mask):
    """
    Calculate 3D displacement from flow data.
    """

    # Function for interpolating coordinates in 3D
    def calcInterpolatedIndex3D(x, y, z, ActInterpFacX, ActInterpFacY, ActInterpFacZ):
        interpX = round((x - 1) * ActInterpFacX + 1)
        interpY = round((y - 1) * ActInterpFacY + 1)
        interpZ = round((z - 1) * ActInterpFacZ + 1)

        # Boundary checks for x
        if interpX < 0:
            interpX = 0
        elif interpX > (interpDispX.shape[0] - 1):
            interpX = interpDispX.shape[0] - 1

        # Boundary checks for y
        if interpY < 0:
            interpY = 0
        elif interpY > (interpDispX.shape[1] - 1):
            interpY = interpDispX.shape[1] - 1

        # Boundary checks for z
        if interpZ < 0:
            interpZ = 0
        elif interpZ > (interpDispX.shape[2] - 1):
            interpZ = interpDispX.shape[2] - 1

        return interpX, interpY, interpZ

    # Read info from header
    nPhases = int(info['CardiacNumberOfImages'])
    SliceThickness = info['SliceThickness']
    dt = float(info['RepetitionTime'])  # ms
    print(f"Repetition time: {dt} ms")

    # Get velocity direction matrix (default to all positive if not specified)
    vel_dir_matrix = info.get('VelocityDirectionMatrix', [1, 1, 1])
    print(f"Velocity direction matrix: {vel_dir_matrix}")

    # Apply velocity direction matrix
    flow_x = flow_x * vel_dir_matrix[0]
    flow_y = flow_y * vel_dir_matrix[1]
    flow_z = flow_z * vel_dir_matrix[2]

    # Get dimensions of the input data
    dimx = flow_x.shape[0]  # height
    dimy = flow_x.shape[1]  # width
    dimz = flow_x.shape[2]  # depth

    # Initialize displacement vectors
    dispVx = np.zeros((dimx, dimy, dimz, nPhases))
    dispVy = np.zeros((dimx, dimy, dimz, nPhases))
    dispVz = np.zeros((dimx, dimy, dimz, nPhases))

    # First, correct the flow data by subtracting mean flow across phases
    # This ensures zero net displacement over the cycle
    flow_x_corrected = flow_x.copy()
    flow_y_corrected = flow_y.copy()
    flow_z_corrected = flow_z.copy()

    # Subtract mean flow across phases (time dimension is the last dimension)
    flow_x_corrected -= np.mean(flow_x_corrected, axis=-1, keepdims=True)
    flow_y_corrected -= np.mean(flow_y_corrected, axis=-1, keepdims=True)
    flow_z_corrected -= np.mean(flow_z_corrected, axis=-1, keepdims=True)

    # Calculate velocity to displacement conversion factor
    # Convert from cm/s to mm/phase (displacement = velocity * time)
    # 10 is to convert cm to mm, dt is in ms so divide by 1000 to get seconds
    conversion_factor = 0.5 * 10 * dt / 1000

    # Calculate differential displacements using corrected flow
    diffDispX = flow_x_corrected * conversion_factor / info['InPlaneResolution'][0]
    diffDispY = flow_y_corrected * conversion_factor / info['InPlaneResolution'][1]
    diffDispZ = flow_z_corrected * conversion_factor / SliceThickness

    print(f"Differential displacement shape: {diffDispX.shape}")

    # Create finer grid for interpolation
    xnew = np.linspace(0, dimx - 1, int(dimx * 1.25))
    ynew = np.linspace(0, dimy - 1, int(dimy * 1.25))
    znew = np.linspace(0, dimz - 1, int(dimz * 1.25))

    X, Y, Z = np.meshgrid(xnew, ynew, znew, indexing='ij')

    # Initialize interpolated displacement arrays
    interpDispX = np.zeros((len(xnew), len(ynew), len(znew), nPhases))
    interpDispY = np.zeros((len(xnew), len(ynew), len(znew), nPhases))
    interpDispZ = np.zeros((len(xnew), len(ynew), len(znew), nPhases))

    print(f"Number of phases: {nPhases}")

    # Perform interpolation for each phase
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

    # Calculate cumulative displacements with Forward/Backward integration
    # First phase has zero displacement
    for iph in range(1, nPhases):
        for ix in range(dimx):
            for iy in range(dimy):
                for iz in range(dimz):
                    if mask[ix, iy, iz]:
                        # Forward displacement: current position + current displacement
                        new_xn = ix + dispVx[ix, iy, iz, iph]
                        new_yn = iy + dispVy[ix, iy, iz, iph]
                        new_zn = iz + dispVz[ix, iy, iz, iph]

                        # Backward displacement: current position - previous displacement
                        new_xnm1 = ix - dispVx[ix, iy, iz, iph - 1]
                        new_ynm1 = iy - dispVy[ix, iy, iz, iph - 1]
                        new_znm1 = iz - dispVz[ix, iy, iz, iph - 1]

                        # Get interpolated indices for forward position
                        inew_xn, inew_yn, inew_zn = calcInterpolatedIndex3D(
                            new_xn,
                            new_yn,
                            new_zn,
                            ActInterpFacX,
                            ActInterpFacY,
                            ActInterpFacZ,
                        )

                        # Get interpolated indices for backward position
                        inew_xnm1, inew_ynm1, inew_znm1 = calcInterpolatedIndex3D(
                            new_xnm1,
                            new_ynm1,
                            new_znm1,
                            ActInterpFacX,
                            ActInterpFacY,
                            ActInterpFacZ,
                        )

                        # Sample velocities at both positions
                        forward_velocity_x = interpDispX[inew_xn, inew_yn, inew_zn, iph]
                        forward_velocity_y = interpDispY[inew_xn, inew_yn, inew_zn, iph]
                        forward_velocity_z = interpDispZ[inew_xn, inew_yn, inew_zn, iph]

                        backward_velocity_x = interpDispX[inew_xnm1, inew_ynm1, inew_znm1, iph - 1]
                        backward_velocity_y = interpDispY[inew_xnm1, inew_ynm1, inew_znm1, iph - 1]
                        backward_velocity_z = interpDispZ[inew_xnm1, inew_ynm1, inew_znm1, iph - 1]

                        # Combine forward and backward velocities (delta displacement)
                        delta_disp_x = backward_velocity_x + forward_velocity_x
                        delta_disp_y = backward_velocity_y + forward_velocity_y
                        delta_disp_z = backward_velocity_z + forward_velocity_z

                        # Calculate total displacement as previous displacement plus delta displacement
                        dispVx[ix, iy, iz, iph] = dispVx[ix, iy, iz, iph - 1] + delta_disp_x
                        dispVy[ix, iy, iz, iph] = dispVy[ix, iy, iz, iph - 1] + delta_disp_y
                        dispVz[ix, iy, iz, iph] = dispVz[ix, iy, iz, iph - 1] + delta_disp_z

    print(f"Displacement shapes: {dispVx.shape}, {dispVy.shape}, {dispVz.shape}")
    #
    # for iph in range(nPhases):
    #     points = (x, y, z)
    #     interp_x_func = scipy.interpolate.RegularGridInterpolator(points, diffDispX[:, :, :, iph], bounds_error=False)
    #     interp_y_func = scipy.interpolate.RegularGridInterpolator(points, diffDispY[:, :, :, iph], bounds_error=False)
    #     interp_z_func = scipy.interpolate.RegularGridInterpolator(points, diffDispZ[:, :, :, iph], bounds_error=False)
    #
    #     pts = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    #
    #     interpDispX[:, :, :, iph] = interp_x_func(pts).reshape(X.shape)
    #     interpDispY[:, :, :, iph] = interp_y_func(pts).reshape(Y.shape)
    #     interpDispZ[:, :, :, iph] = interp_z_func(pts).reshape(Z.shape)
    #
    # ActInterpFacX = len(xnew) / len(x)
    # ActInterpFacY = len(ynew) / len(y)
    # ActInterpFacZ = len(znew) / len(z)
    #
    # print(f"ActInterpFacX: {ActInterpFacX}, ActInterpFacY: {ActInterpFacY}, ActInterpFacZ: {ActInterpFacZ}")
    #
    # # Calculate cumulative displacements
    # # First phase has zero displacement
    # for iph in range(1, nPhases):
    #     for ix in range(dimx):
    #         for iy in range(dimy):
    #             for iz in range(dimz):
    #                 if mask[ix, iy, iz]:
    #                     # Get position at previous time step including displacement
    #                     new_xn = ix + dispVx[ix, iy, iz, iph - 1]
    #                     new_yn = iy + dispVy[ix, iy, iz, iph - 1]
    #                     new_zn = iz + dispVz[ix, iy, iz, iph - 1]
    #
    #                     # Get interpolated indices for this position
    #                     inew_xn, inew_yn, inew_zn = calcInterpolatedIndex3D(
    #                         new_xn,
    #                         new_yn,
    #                         new_zn,
    #                         ActInterpFacX,
    #                         ActInterpFacY,
    #                         ActInterpFacZ,
    #                     )
    #
    #                     # Get velocity at this new position from the interpolated grid
    #                     current_velocity_x = interpDispX[inew_xn, inew_yn, inew_zn, iph]
    #                     current_velocity_y = interpDispY[inew_xn, inew_yn, inew_zn, iph]
    #                     current_velocity_z = interpDispZ[inew_xn, inew_yn, inew_zn, iph]
    #
    #                     # Calculate total displacement as previous displacement plus new displacement
    #                     dispVx[ix, iy, iz, iph] = dispVx[ix, iy, iz, iph - 1] + current_velocity_x
    #                     dispVy[ix, iy, iz, iph] = dispVy[ix, iy, iz, iph - 1] + current_velocity_y
    #                     dispVz[ix, iy, iz, iph] = dispVz[ix, iy, iz, iph - 1] + current_velocity_z
    #
    # print(f"Displacement shapes: {dispVx.shape}, {dispVy.shape}, {dispVz.shape}")

    # Convert displacements to physical units (mm)
    dispVxi_mm = info['InPlaneResolution'][0] * dispVx
    dispVyi_mm = info['InPlaneResolution'][1] * dispVy
    dispVzi_mm = SliceThickness * dispVz

    return dispVxi_mm.astype(np.float32), dispVyi_mm.astype(np.float32), dispVzi_mm.astype(np.float32)


def calc_strain_3d(dispX, dispY, dispZ, mask, fx=None, fy=None, fz=None):
    """
    Calculate 3D strain tensor and eigenvalues for multiple phases.
    """

    def strain3D(dispX_3D, dispY_3D, dispZ_3D):
        """Calculate strain tensor for a single phase"""
        dimx, dimy, dimz = dispX_3D.shape

        # Initialize strain tensor (3x3 for each point)
        E = np.zeros((dimx, dimy, dimz, 3, 3))

        # Calculate all partial derivatives using 3D Savitzky-Golay filter
        Uxx = sgolay3d(dispX_3D, window_size=13, order=4, derivative='x')
        Uxy = sgolay3d(dispX_3D, window_size=13, order=4, derivative='y')
        Uxz = sgolay3d(dispX_3D, window_size=13, order=4, derivative='z')

        Uyx = sgolay3d(dispY_3D, window_size=13, order=4, derivative='x')
        Uyy = sgolay3d(dispY_3D, window_size=13, order=4, derivative='y')
        Uyz = sgolay3d(dispY_3D, window_size=13, order=4, derivative='z')

        Uzx = sgolay3d(dispZ_3D, window_size=13, order=4, derivative='x')
        Uzy = sgolay3d(dispZ_3D, window_size=13, order=4, derivative='y')
        Uzz = sgolay3d(dispZ_3D, window_size=13, order=4, derivative='z')

        # Calculate strain tensor for each point
        for ix in range(dimx):
            for iy in range(dimy):
                for iz in range(dimz):
                    # The displacement gradient
                    Ugrad = np.array([
                        [Uxx[ix, iy, iz], Uxy[ix, iy, iz], Uxz[ix, iy, iz]],
                        [Uyx[ix, iy, iz], Uyy[ix, iy, iz], Uyz[ix, iy, iz]],
                        [Uzx[ix, iy, iz], Uzy[ix, iy, iz], Uzz[ix, iy, iz]]
                    ])

                    # The (inverse) deformation gradient
                    Finv = np.eye(3) - Ugrad

                    # The 3D Eulerian strain tensor
                    e = 0.5 * (np.eye(3) - Finv @ Finv.T)

                    # Store tensor in the output matrix
                    E[ix, iy, iz, :, :] = e

        return E

    # Get dimensions
    dimx, dimy, dimz = dispX.shape[:3]
    n_phases = dispX.shape[3] if len(dispX.shape) > 3 else 1

    # Initialize eigenvalue array (3 eigenvalues for each point and phase)
    Eig_v = np.zeros((dimx, dimy, dimz, 3, n_phases))

    # Process each phase separately
    for iph in range(n_phases):
        if n_phases > 1:
            # Extract the displacement field for this phase
            dispX_phase = dispX[:, :, :, iph]
            dispY_phase = dispY[:, :, :, iph]
            dispZ_phase = dispZ[:, :, :, iph]
        else:
            # If there's only one phase, use the input directly
            dispX_phase = dispX
            dispY_phase = dispY
            dispZ_phase = dispZ

        # Calculate the 3D strain tensor for this phase
        strain_tensor = strain3D(dispX_phase, dispY_phase, dispZ_phase)

        # Calculate eigenvalues at each point
        for ix in range(dimx):
            for iy in range(dimy):
                for iz in range(dimz):
                    # Check if this point is within the mask
                    # Adapt the condition based on mask dimensions
                    if mask.ndim == 3:  # 3D mask
                        mask_condition = mask[ix, iy, iz]
                    elif mask.ndim == 2:  # 2D mask
                        mask_condition = mask[ix, iy]
                    else:  # Handle other cases
                        mask_condition = True  # Default to processing all points

                    if mask_condition:
                        # Calculate eigenvalues and sort them
                        eigenvalues = LA.eigvals(strain_tensor[ix, iy, iz, :, :])
                        # Sort in descending order (e1: stretching, e2: intermediate, e3: compression)
                        Eig_v[ix, iy, iz, :, iph] = np.sort(eigenvalues)[::-1]

    # If there's only one phase, remove the phase dimension for compatibility
    if n_phases == 1:
        Eig_v = Eig_v[:, :, :, :, 0]

    return Eig_v


def sum_strain_3d(Eig_v, mask, info):
    """
    Summarize strain values by calculating median values within mask.
    """
    nPhases = int(info['CardiacNumberOfImages'])
    e1_Line = np.zeros(nPhases)  # First principal strain (largest eigenvalue)
    e2_Line = np.zeros(nPhases)  # Second principal strain
    e3_Line = np.zeros(nPhases)  # Third principal strain (smallest eigenvalue)

    for i_ph in range(nPhases):
        #print(nPhases)
        E1 = Eig_v[:, :, :, 0, i_ph]  # First principal strain
        E2 = Eig_v[:, :, :, 1, i_ph]  # Second principal strain
        E3 = Eig_v[:, :, :, 2, i_ph]  # Third principal strain

        # Calculate median of non-zero values within the mask
        masked_E1 = E1[mask]  # Result shape: (n_masked_voxels, z)
        masked_E2 = E2[mask]
        masked_E3 = E3[mask]

        if len(masked_E1) > 0:
            e1_Line[i_ph] = np.nanmedian(masked_E1[masked_E1 != 0])
        if len(masked_E2) > 0:
            e2_Line[i_ph] = np.nanmedian(masked_E2[masked_E2 != 0])
        if len(masked_E3) > 0:
            e3_Line[i_ph] = (np.nanmedian(masked_E3[masked_E3 != 0]))

    # Maximum values
    e1_max = np.round(np.nanmax(e1_Line), 3)  # First principal strain (stretching)
    e2_max = np.round(np.nanmax(e2_Line), 3)  # Second principal strain
    e3_max = np.round(np.nanmax(e3_Line), 3)  # Third principal strain (compression)

    return e1_Line, e1_max, e2_Line, e2_max, e3_Line, e3_max


def load_config(config_path):
    """Load and process JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate 3D strain from velocity data.')

    # Input files
    parser.add_argument('--data-path', required=True, help='Path to directory containing velocity data files')
    parser.add_argument('--prefix', help='Prefix of velocity data files (optional)')

    # Mask files
    parser.add_argument('--mask', help='Path to mask file (.npy) (optional)')
    parser.add_argument('--roi-mask', help='Path to ROI mask file (.npy) (optional)')

    # Output files
    parser.add_argument('--output-eig', default='Eig_v_output.npy', help='Output eigenvalue file (.npy)')
    parser.add_argument('--output-disp-x', help='Output displacement x file (.npy) (optional)')
    parser.add_argument('--output-disp-y', help='Output displacement y file (.npy) (optional)')
    parser.add_argument('--output-disp-z', help='Output displacement z file (.npy) (optional)')
    parser.add_argument('--output-plot', help='Output strain plot file (.png) (optional)')

    # Parameters
    parser.add_argument('--in-plane-resolution', type=float, default=1.5, help='In-plane resolution in mm')
    parser.add_argument('--slice-thickness', type=float, default=1.5, help='Slice thickness in mm')
    parser.add_argument('--repetition-time', type=float, default=6.7, help='Repetition time in ms')
    parser.add_argument('--slice-index', type=int, help='Slice index for 2D analysis (optional)')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots')
    parser.add_argument('--config', required=True, help='Path to JSON configuration file')

    args = parser.parse_args()

    # Load combined data file
    data_path = args.data_path
    prefix = args.prefix or "DATA"

    # Find the new single file
    processed_file_pattern = os.path.join(data_path, f"{prefix}*_processed_data.npy")
    processed_files = glob.glob(processed_file_pattern)

    if not processed_files:
        raise FileNotFoundError(f"Could not find processed data file in {data_path} with prefix {prefix}")

    processed_file = processed_files[0]
    print(f"Found processed data file: {os.path.basename(processed_file)}")

    # Load the dictionary from the .npy file
    processed_data = np.load(processed_file, allow_pickle=True).item()

    # Extract the velocities and mask
    velocities = processed_data['velocities']
    mask_4d = processed_data['mask']
    mask = mask_4d[..., 0]

    # Split the velocities array into x, y, and z components
    flow_x = velocities[..., 0]
    flow_y = velocities[..., 1]
    flow_z = velocities[..., 2]

    print(f"Velocity data shape: {velocities.shape}")
    print(f"Mask data shape: {mask.shape}")

    # Load configuration from JSON file
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Update config values with actual data
    info = {}
    for key, value in config.items():
        if key == 'CardiacNumberOfImages':
            info[key] = velocities.shape[3]
        elif key == 'NofPoints':
            info[key] = velocities.shape[0]
        elif key == 'VelocityDirectionMatrix':
            if len(value) != 3:
                print("Warning: VelocityDirectionMatrix should have exactly 3 elements. Using default [1,1,1].")
                info[key] = [1, 1, 1]
            else:
                info[key] = value
        elif isinstance(value, list):
            info[key] = value
        else:
            # --- SAFE CONVERSION BLOCK ---
            if isinstance(value, str):
                try:
                    info[key] = float(value)
                except ValueError:
                    info[key] = value  # Keep as string (e.g. BART text)
            else:
                info[key] = value


    print("Configuration parameters:")
    for key, val in info.items():
        print(f"  {key}: {val}")

    # Now you can delete your second "for key, value in info.items()" loop
    # because the block above already handled the conversion.
    print("Calculating displacements...")
    dispVxi, dispVyi, dispVzi = calc_disp_3d(flow_x, flow_y, flow_z, info, mask)

    # Save displacement data if requested
    if args.output_disp_x:
        print(f"Saving displacement x to: {args.output_disp_x}")
        np.save(args.output_disp_x, dispVxi)
    if args.output_disp_y:
        print(f"Saving displacement y to: {args.output_disp_y}")
        np.save(args.output_disp_y, dispVyi)
    if args.output_disp_z:
        print(f"Saving displacement z to: {args.output_disp_z}")
        np.save(args.output_disp_z, dispVzi)

    # Calculate strain
    print("Calculating strain...")
    Eig_v = calc_strain_3d(dispVxi, dispVyi, dispVzi, mask)

    # Save eigenvalue data
    print(f"Saving eigenvalues to: {args.output_eig}")
    if os.path.isabs(args.output_eig):
        output_eig_path = args.output_eig
    else:
        output_eig_path = os.path.join(args.data_path, args.output_eig)
    np.save(output_eig_path, Eig_v)

    # Process ROI mask if provided
    if args.roi_mask:
        print(f"Loading ROI mask from: {args.roi_mask}")
        roi_mask = np.load(args.roi_mask)
        use_mask = roi_mask
    else:
        use_mask = mask

    # If slice index is provided, use only that slice
    if args.slice_index is not None:
        print(f"Using slice index: {args.slice_index}")
        if len(Eig_v.shape) == 5:  # 3D + 3 eigenvalues + phases
            slice_eig = Eig_v[args.slice_index, :, :, :, :]
            if len(use_mask.shape) == 4:  # 4D mask
                slice_mask = use_mask[args.slice_index, :, :, :]
            else:
                slice_mask = use_mask
        else:
            slice_eig = Eig_v
            slice_mask = use_mask

        e1_Line, e1_max, e2_Line, e2_max, e3_Line, e3_max = sum_strain_3d(slice_eig, slice_mask, info)
    else:
        # Process full volume
        e1_Line, e1_max, e2_Line, e2_max, e3_Line, e3_max = sum_strain_3d(Eig_v, use_mask, info)

    # Display results
    print(f"First principal strain max: {e1_max}")
    print(f"Second principal strain max: {e2_max}")
    print(f"Third principal strain max: {e3_max}")

    # Create and save plots if not disabled
    if not args.no_display or args.output_plot:
        plt.figure(figsize=(10, 6))
        for data, label, color in zip([e1_Line, e2_Line, e3_Line],
                                      ['e1 (stretching)', 'e2 (intermediate)', 'e3 (compression)'],
                                      ['blue', 'green', 'red']):
            plt.plot(data, label=f"{label}", color=color)

        plt.xlabel('Cardiac Phase')
        plt.ylabel('Strain')
        plt.title('Principal Strains')
        plt.legend()
        plt.grid(True)

        if args.output_plot:
            print(f"Saving plot to: {args.output_plot}")
            plt.savefig(args.output_plot, dpi=300)

        if not args.no_display:
            plt.show()



if __name__ == "__main__":
    main()