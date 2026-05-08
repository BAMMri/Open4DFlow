import os
import numpy as np
import sys
from tqdm import tqdm

# Set GPU environment variables for BART
os.environ['BART_CUDA_GPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0, change if you have multiple GPUs

# Add your bart path
sys.path.append('/home/marta/Projects/bart_projects/bart/python')

import bart
from bart import cfl

bart_orig = bart.bart


def bart_print(nargout, cmd, *args):
    print("Executing bart", cmd)
    return bart_orig(nargout, cmd, *args)


bart.bart = bart_print


def reconstruct_and_process_all(data_dir, venc_value=20, use_gpu=True, ecalib_r='20:20', compressed_coils=0,
                                joint_recon=False):

    # Set GPU usage
    if use_gpu:
        print("GPU acceleration enabled")
        os.environ['BART_CUDA_GPU'] = '1'
        gpu_flag = '-g'
    else:
        print("GPU acceleration disabled")
        os.environ['BART_CUDA_GPU'] = '0'
        gpu_flag = ''

    print(f'{ecalib_r=}')

    # Find all .cfl/.hdr pairs
    file_bases = [
        os.path.splitext(f)[0]
        for f in os.listdir(data_dir)
        if f.endswith('.cfl') and os.path.exists(os.path.join(data_dir, os.path.splitext(f)[0] + '.hdr'))
    ]
    if not file_bases:
        print("No .cfl/.hdr data files found in the directory. Exiting.")
        return

    for file_base in file_bases:
        cfl_path = os.path.join(data_dir, file_base)
        print(f"Reading: {cfl_path}")
        ksp = cfl.readcfl(cfl_path)

        if compressed_coils > 0:
            ksp = bart.bart(1, f'cc -p{compressed_coils}', ksp)

        # Infer dimensions
        num_x, num_y, num_slices = ksp.shape[0:3]
        num_coils = ksp.shape[3]
        num_cardiac_phases = ksp.shape[6]
        num_vencs = ksp.shape[11]
        print(f"Shape: {ksp.shape}, num_vencs: {num_vencs}, num_cardiac_phases: {num_cardiac_phases}")

        # Reconstruct images for all vencs and cardiac phases
        recon = np.zeros((num_x, num_y, num_slices, num_cardiac_phases, num_vencs), dtype=np.complex64)
        print('venc_value', venc_value)
        def recon_separate():
            for venc in range(num_vencs):
                for phase in range(num_cardiac_phases):
                    print(f"Processing venc {venc + 1}/{num_vencs}, phase {phase + 1}/{num_cardiac_phases}")

                    all_ksp_venc_phase = ksp[:, :, :, :, 0, 0, phase, 0, 0, 0, 0, venc]
                    all_ksp_esprit = ksp[:, :, :, :, 0, 0, phase, 0, 0, 0, 0, 0]

                    # ESPIRiT coil sensitivity estimation
                    if use_gpu:
                        sensitivities = bart.bart(1, f"ecalib -g -c0 -m1 -r{ecalib_r}", all_ksp_esprit)
                    else:
                        sensitivities = bart.bart(1, f"ecalib -c0 -m1 -r{ecalib_r}", all_ksp_esprit)

                    l1_wav_reg = 0.005

                    # PICS reconstruction
                    if use_gpu:
                        image_l1_tv_wav = bart.bart(
                            1,
                            f"pics -g -R W:7:0:{l1_wav_reg} -e -i 20 -S -d5",
                            all_ksp_venc_phase,
                            sensitivities
                        )
                    else:
                        image_l1_tv_wav = bart.bart(
                            1,
                            f"pics -R W:7:0:{l1_wav_reg} -e -i 20 -S -d5",
                            all_ksp_venc_phase,
                            sensitivities
                        )

                    recon[:, :, :, phase, venc] = image_l1_tv_wav.squeeze()

        def recon_joint():
            nonlocal recon
            ksp_flattened = np.reshape(ksp, (
            num_x, num_y, num_slices, num_coils, 1, 1, num_cardiac_phases * num_vencs, 1, 1, 1, 1, 1))
            ksp_espirit = np.sum(ksp_flattened, axis=6)

            if use_gpu:
                gpu_flag = '-g'
            else:
                gpu_flag = ''

            sensitivities = bart.bart(1, f"ecalib {gpu_flag} -c0 -m1 -r{ecalib_r}", ksp_espirit)

            for venc in tqdm(range(num_vencs)):
                l1_wav_reg = 0.009
                tv_reg = 0.0001

                ksp_to_recon = np.zeros((num_x, num_y, num_slices, num_coils, 1, 1, num_cardiac_phases, 1, 1, 1, 1, 1),
                                        np.complex64)
                ksp_to_recon[:, :, :, :, 0, 0, :, 0, 0, 0, 0, 0] = ksp[:, :, :, :, 0, 0, :, 0, 0, 0, 0, venc]

                print("Shape to recon", ksp_to_recon.shape)

                image_l1_tv_wav = bart.bart(
                    1,
                    f"pics {gpu_flag} -R W:7:0:{l1_wav_reg} -R T:64:0:{tv_reg} -e -i 20 -S -d2",
                    ksp_to_recon,
                    sensitivities
                )
                recon[:, :, :, :, venc] = image_l1_tv_wav.squeeze()

        if joint_recon:
            recon_joint()
        else:
            recon_separate()

        # Save reconstructed images
        out_file = os.path.join(data_dir, f"DATA_{os.path.basename(file_base)}.npy")
        np.save(out_file, recon)
        print(f"Saved reconstructed image: {out_file}")

        arr = recon
        print(f"Processing velocities, arr shape: {arr.shape}")

        if arr.shape[-1] < 4:
            print(f"ERROR: Not enough VENC encodings to compute x, y, z velocities for {file_base}!")
            continue

        # Calculate phase differences for each component
        # phase_diff_x = np.angle(arr[..., 1]) - np.angle(arr[..., 0])  # x-component
        # phase_diff_y = np.angle(arr[..., 2]) - np.angle(arr[..., 0])  # y-component
        # phase_diff_z = np.angle(arr[..., 3]) - np.angle(arr[..., 0])  # z-component
        #
        phase_diff_x = np.angle(arr[..., 1] * np.conj(arr[..., 0]))
        phase_diff_y = np.angle(arr[..., 2] * np.conj(arr[..., 0]))
        phase_diff_z = np.angle(arr[..., 3] * np.conj(arr[..., 0]))

        # Combine phase differences into single array
        phase_diffs = np.stack([phase_diff_x, phase_diff_y, phase_diff_z], axis=-1)
        # print(f"Max phase_diff_x: {np.max(phase_diff_x)}")
        # print(f"Min phase_diff_x: {np.min(phase_diff_x)}")

        # Convert phase differences to velocities using VENC
        velocities = (phase_diffs / np.pi) * venc_value  # Shape: (X, Y, Z, Time, 3)
        # print(f"Max vel: {np.max(velocities)}")
        # print(f"Min vel: {np.min(velocities)}")
        # print(f"Mean vel (in vessel): {np.mean(velocities[mask])}")
        # Create very rough mask based on magnitude data
        mag_data = np.abs(arr[..., 0])  # Use reference magnitude
        mag_data_alldirs = np.abs(arr)
        mask = mag_data > 0.10 * np.max(mag_data)
        # #mask = mag_data > 0.1 * np.max(mag_data)
        # for phase in range(num_cardiac_phases):
        #     for axis in range(3):  # x, y, z
        #         # Calculate mean velocity in masked region
        #         mean_offset = np.mean(velocities[:, :, :, phase, axis][mask[:, :, :, phase]])
        #         # Subtract from all pixels (assumes net flow through ROI should be zero)
        #         velocities[:, :, :, phase, axis] -= mean_offset
        # # Apply mask to velocities
        velocities = velocities * mask[..., np.newaxis]

        # Save combined data
        name = os.path.splitext(os.path.basename(out_file))[0]
        data_to_save = {
            'velocities': velocities,
            'mask': mask,
            'magnitude': mag_data_alldirs
        }
        np.save(os.path.join(data_dir, f'{name}_processed_data.npy'), data_to_save)
        print(f"Processed and saved combined velocity data and mask for {name}")


def check_gpu_availability():
    """Check if GPU is available and BART was compiled with CUDA support."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected")
            return True
        else:
            print("No NVIDIA GPU detected")
            return False
    except FileNotFoundError:
        print("nvidia-smi not found. No GPU support detected.")
        return False




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct and process 4D flow MRI data.")
    parser.add_argument("data_dir", type=str, help="Path to the data directory containing .cfl/.hdr files.")
    parser.add_argument("--venc", type=float, default=20, help="VENC value in cm/s.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration.")
    parser.add_argument("--ecalib-r", type=str, default="12:9",
                        help='Value to use for the -r argument in ecalib (e.g., "12:9")')
    parser.add_argument("--cc", type=int, default=0,
                        help='Number of compressed coils for the processing. Default: no compression')
    parser.add_argument("--joint", action="store_true", help='Perform joint reconstruction of phases and velocities')
    args = parser.parse_args()

    # Check GPU availability
    gpu_available = check_gpu_availability()
    use_gpu = gpu_available and not args.no_gpu

    if args.no_gpu:
        print("GPU acceleration disabled by user")
    elif not gpu_available:
        print("GPU not available, falling back to CPU")

    reconstruct_and_process_all(args.data_dir, venc_value=args.venc, use_gpu=use_gpu, ecalib_r=args.ecalib_r,
                                compressed_coils=args.cc, joint_recon=args.joint)