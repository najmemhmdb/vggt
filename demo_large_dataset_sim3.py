import os
import glob
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import predictions_to_glb
from demo_viser import viser_wrapper
from my_alignment_utils import umeyama_sim3

# Argument parser
parser = argparse.ArgumentParser(description="VGGT demo with viser for large datasets using overlap-based alignment")
parser.add_argument(
    "--image_folder", type=str, default="../data/new_office/frames", help="Path to folder containing all images"
)
parser.add_argument("--use_point_map", default=False, action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--window_size", type=int, default=20, help="Number of frames per processing window")
parser.add_argument("--overlap", type=int, default=15, help="Number of overlapping frames between windows")
parser.add_argument("--save_results", type=str, default="glbscene_50_20_15_sim3.glb", help="Path to save unified results (.npz)")


def to_4x4(extrinsic_3x4):  
    """Convert 3x4 extrinsic matrix to 4x4 homogeneous matrix."""
    extrinsic_4x4 = np.eye(4)
    extrinsic_4x4[:3, :] = extrinsic_3x4
    return extrinsic_4x4


def to_3x4(extrinsic_4x4):
    """Convert 4x4 homogeneous matrix to 3x4 extrinsic matrix."""
    return extrinsic_4x4[:3, :]


def get_camera_centers(extrinsics_4x4):
    R = extrinsics_4x4[:, :3, :3]
    t = extrinsics_4x4[:, :3, 3:]
    C = -np.matmul(np.transpose(R, (0, 2, 1)), t)  # -Rᵀ t
    return C.squeeze(-1)

def transform_poses_to_reference(extrinsics_window, transformation_matrix):
    """
    Transform poses in a window to be relative to the reference coordinate system.
    
    Args:
        extrinsics_window (np.ndarray): Window extrinsics of shape (S, 3, 4)
        transformation_matrix (np.ndarray): Noise-free Transformation (4, 4)
    
    Returns:
        np.ndarray: Transformed extrinsics relative to reference
    """
    T_align_inv = closed_form_inverse_se3(transformation_matrix[None])[0]
    transformed_extrinsics = []

    for i in range(extrinsics_window.shape[0]):
        current_4x4 = to_4x4(extrinsics_window[i])
        transformed_4x4 = current_4x4 @ T_align_inv
        transformed_extrinsics.append(to_3x4(transformed_4x4))

    return np.stack(transformed_extrinsics)


def calculate_transformation(prev_overlaps, curr_overlaps):
    """
    Calculate noise-free transformation between two windows
    Args:
        prev_overlaps (np.ndarray): Previous overlaps of shape (S, 3, 4)
        curr_overlaps (np.ndarray): Current overlaps of shape (S, 3, 4)
    Returns:
        np.ndarray: noise-free transformation of shape (3, 4)
    """
    # camera centres (world coords) are rows 3 of inv(E)
    C_prev = get_camera_centers(np.stack(prev_overlaps))
    C_curr = get_camera_centers(np.stack(curr_overlaps))
    print(C_prev.shape, C_curr.shape)
    S, R, t = umeyama_sim3(C_curr, C_prev)
    T_align = np.eye(4)
    T_align[:3, :3] = S*R
    T_align[:3, 3] = t

    # T_align = closed_form_inverse_se3(prev_overlaps[0][None])[0]
    return T_align


# def extrinsic_averaging(prev_ext, curr_ext):
#     """
#     Calculate average of previous window prediction for a frame and the current one
#     Args:
#         prev_ext (np.ndarray): Previous Extrinsic Prediction (3, 4)
#         curr_ext (np.ndarray): Current Extrinsic Prediction (3, 4)
#     Returns:
#         np.ndarray: Average exitrinsc of shape (3, 4)
#     """
#     rotations = []
#     r_prev = R.from_matrix(prev_ext[:3, :3])
#     rotations.append(r_prev.as_euler('zyx', degrees=True))
#     r_curr = R.from_matrix(curr_ext[:3, :3])
#     rotations.append(r_curr.as_euler('zyx', degrees=True))


#     r_total = R.from_euler('zyx', rotations, degrees=True)
#     rotation_mean = r_total.mean().as_matrix()

#     translation_mean = np.mean([prev_ext[:3, 3], curr_ext[:3, 3]], axis=0)

#     ext_mean = np.eye(4)
#     ext_mean[:3, :3] = rotation_mean
#     ext_mean[:3, 3] = translation_mean

#     return to_3x4(ext_mean)


# def relative_transform(T1, T2):
#     """Compute relative transformation T = E2 @ inv(E1)"""
#     return T2 @ np.linalg.inv(T1)


# def compare_relative_transforms(E5_w0, E6_w0, E5_w1, E6_w1):
#     """Compare relative transforms between (E5→E6) in both windows"""
#     T_5to6_w0 = relative_transform(E5_w0, E6_w0)
#     T_5to6_w1 = relative_transform(E5_w1, E6_w1)

#     # Compare translation
#     t0 = T_5to6_w0[:3, 3]
#     t1 = T_5to6_w1[:3, 3]
#     t_error = np.linalg.norm(t0 - t1)

#     # Compare rotation
#     R0 = T_5to6_w0[:3, :3]
#     R1 = T_5to6_w1[:3, :3]
#     R_diff = R.from_matrix(R0 @ R1.T)
#     r_error = R_diff.magnitude() * (180.0 / np.pi)

#     print(f"Translation error: {t_error:.6f} m")
#     print(f"Rotation error:    {r_error:.4f} degrees")
#     return t_error, r_error


def process_large_dataset_windowed(image_paths, window_size=70, overlap=20, device="cuda"):
    """
    Process a large dataset of images using VGGT in overlapping windows with overlap-based alignment.
    
    Args:
        image_paths (list): List of paths to all images
        window_size (int): Number of frames per window (default: 70)
        overlap (int): Number of overlapping frames between windows (default: 20)
        device (str): Device to run inference on
    
    Returns:
        dict: Unified results containing all predictions
    """
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    
    # Sort image paths to ensure correct order
    image_paths = sorted(image_paths)
    total_frames = len(image_paths)
    print(f"Processing {total_frames} frames in windows of {window_size} with {overlap} overlap")
    
    # Calculate window positions
    stride = window_size - overlap
    window_starts = list(range(0, total_frames - window_size + 1, stride))
    
    # Ensure last window doesn't exceed bounds
    if window_starts[-1] + window_size < total_frames:
        window_starts.append(window_starts[-1]+stride)
        
    print('window_starts', window_starts)
    print(f"Will process {len(window_starts)} windows")
    
    # Storage for unified results
    unified_results = {
        'extrinsics': {},
        'intrinsics': {},
        'images': {},
        'world_points': {},
        'world_points_conf': {},
        'depth': {},
        'depth_conf': {}
    }
    # Process each window
    for window_idx, start_idx in enumerate(tqdm(window_starts, desc="Processing windows")):
        end_idx = min(start_idx + window_size, total_frames)
        window_image_paths = image_paths[start_idx:end_idx]
        
        print(f"\nWindow {window_idx + 1}/{len(window_starts)}: frames {start_idx}-{end_idx-1}")
        

        # Load and preprocess images for current window
        images = load_and_preprocess_images(window_image_paths).to(device)
        print(f"Loaded window images shape: {images.shape}")
        
        # Run VGGT inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        
        # Convert pose encoding to matrices
        extrinsic_window, intrinsic_window = pose_encoding_to_extri_intri(
            predictions["pose_enc"], images.shape[-2:]
        )
        
        # Convert to numpy and remove batch dimension
        extrinsic_window = extrinsic_window.squeeze(0).cpu().numpy()  # (S, 3, 4)
        intrinsic_window = intrinsic_window.squeeze(0).cpu().numpy()   # (S, 3, 3)
        images_np = images.cpu().numpy()  # (S, 3, H, W)
        world_points_window = predictions["world_points"].squeeze(0).cpu().numpy()  # (S, H, W, 3)
        world_points_conf = predictions["world_points_conf"].squeeze(0).cpu().numpy()  # (S, H, W)
        depth_maps = predictions["depth"].squeeze(0).cpu().numpy()     # (S, H, W, 1)
        depth_confs = predictions["depth_conf"].squeeze(0).cpu().numpy() # (S, H, W)
        transformed_extrinsics = extrinsic_window
        # Handle alignment based on window
        if window_idx == 0:
            transformed_extrinsics = extrinsic_window
       
        if window_idx > 0:
            common_frame_idx = start_idx
            if common_frame_idx not in unified_results['extrinsics']:
                raise ValueError(f"Common frame {common_frame_idx} missing in previous window")

            prev_overlaps = []
            for k, v in unified_results['extrinsics'].items():
                if k >= common_frame_idx:
                    prev_overlaps.append(to_4x4(v.copy()))
            curr_overlaps = []
            for i in range(overlap):
                curr_overlaps.append(to_4x4(extrinsic_window[i].copy()))
            # for o_i in range(overlap-1):
            #     compare_relative_transforms(prev_overlaps[o_i], prev_overlaps[o_i+1], curr_overlaps[o_i], curr_overlaps[o_i+1])

            T_align = calculate_transformation(prev_overlaps, curr_overlaps)
            transformed_extrinsics = transform_poses_to_reference(
                extrinsic_window, T_align
            )
   
        for i in range(len(transformed_extrinsics)):
            global_frame_idx = start_idx + i
            # global_frame_idx = start_idx + i + (window_idx * overlap)
            # if global_frame_idx not in unified_results['extrinsics']:
            unified_results['extrinsics'][global_frame_idx] = transformed_extrinsics[i]
            unified_results['intrinsics'][global_frame_idx] = intrinsic_window[i]
            unified_results['images'][global_frame_idx] = images_np[i]
            unified_results['world_points'][global_frame_idx] = world_points_window[i]
            unified_results['world_points_conf'][global_frame_idx] = world_points_conf[i]
            unified_results['depth'][global_frame_idx] = depth_maps[i]
            unified_results['depth_conf'][global_frame_idx] = depth_confs[i]
            # else:
                # unified_results['extrinsics'][global_frame_idx] = extrinsic_averaging(unified_results['extrinsics'][global_frame_idx], 
                #                                                                       transformed_extrinsics[i])
                
        # Clear GPU memory
        del predictions, images
        torch.cuda.empty_cache()
        

    # Convert dictionaries to arrays (sorted by frame index)
    frame_indices = sorted(unified_results['extrinsics'].keys())
    
    final_results = {
        'images': np.stack([unified_results['images'][i] for i in frame_indices]),
        'extrinsic': np.stack([unified_results['extrinsics'][i] for i in frame_indices]),
        'intrinsic': np.stack([unified_results['intrinsics'][i] for i in frame_indices]),
        'world_points': np.stack([unified_results['world_points'][i] for i in frame_indices]),
        'world_points_conf': np.stack([unified_results['world_points_conf'][i] for i in frame_indices]),
        'depth': np.stack([unified_results['depth'][i] for i in frame_indices]),
        'depth_conf': np.stack([unified_results['depth_conf'][i] for i in frame_indices])
    }
    
    print(f"\nUnified processing complete with overlap-based alignment!")
    print(f"Final results shape:")
    print(f"- Images: {final_results['images'].shape}")
    print(f"- Extrinsics: {final_results['extrinsic'].shape}")
    print(f"- Intrinsics: {final_results['intrinsic'].shape}")
    print(f"- World points: {final_results['world_points'].shape}")
    print(f"- Depth: {final_results['depth'].shape}")
    
    return final_results



def main():
    """
    Main function for processing large datasets with VGGT using overlap-based alignment and visualizing with viser.
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get all image paths
    print(f"Loading images from {args.image_folder}...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(args.image_folder, ext.upper())))
    
    image_paths = sorted(image_paths)
    
    print(f"Found {len(image_paths)} images")
    new_image_paths = []
    for i in range(0, len(image_paths), 30):
        new_image_paths.append(image_paths[i])
    image_paths = new_image_paths[0:50]
    print(f"Found {len(image_paths)} images")


    if len(image_paths) == 0:
        raise ValueError("No images found. Check your image folder path.")

    # Process the dataset in windows with overlap-based alignment
    print("Processing large dataset in windows with overlap-based alignment...")
    predictions = process_large_dataset_windowed(
        image_paths=image_paths,
        window_size=args.window_size,
        overlap=args.overlap,
        device=device
    )

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")
    # viser_server = viser_wrapper(
    #     predictions,
    #     port=args.port,
    #     init_conf_threshold=args.conf_threshold,
    #     use_point_map=args.use_point_map,
    #     background_mode=args.background_mode,
    #     mask_sky=args.mask_sky,
    #     image_folder=args.image_folder,
    # )
    glbfile = os.path.join(
        args.save_results)
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=50.0,
        filter_by_frames="all",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        target_dir=None,
        prediction_mode="Predicted Pointmap", 
        use_point_map=args.use_point_map)
    glbscene.export(file_obj=glbfile)
    print("Visualization complete")


if __name__ == "__main__":
    main()
