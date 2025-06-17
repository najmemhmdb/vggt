import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from demo_viser import viser_wrapper


# Argument parser
parser = argparse.ArgumentParser(description="VGGT demo with viser for large datasets using overlap-based alignment")
parser.add_argument(
    "--image_folder", type=str, default="../data/new_office/frames", help="Path to folder containing all images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument("--window_size", type=int, default=5, help="Number of frames per processing window")
parser.add_argument("--overlap", type=int, default=1, help="Number of overlapping frames between windows")
parser.add_argument("--save_results", type=str, default=None, help="Path to save unified results (.npz)")


def to_4x4(extrinsic_3x4):
    """Convert 3x4 extrinsic matrix to 4x4 homogeneous matrix."""
    extrinsic_4x4 = np.eye(4)
    extrinsic_4x4[:3, :] = extrinsic_3x4
    return extrinsic_4x4


def to_3x4(extrinsic_4x4):
    """Convert 4x4 homogeneous matrix to 3x4 extrinsic matrix."""
    return extrinsic_4x4[:3, :]


def extrinsic_to_pose(extrinsic):
    """Convert camera extrinsic to camera pose (position in world) using proper SE3 inverse"""
    return closed_form_inverse_se3(to_4x4(extrinsic)[None])[0]


def transform_poses_to_reference(extrinsics_window, reference_extrinsic):
    """
    Transform poses in a window to be relative to the reference coordinate system.
    
    Args:
        extrinsics_window (np.ndarray): Window extrinsics of shape (S, 3, 4)
        reference_extrinsic (np.ndarray): Reference extrinsic of shape (3, 4)
    
    Returns:
        np.ndarray: Transformed extrinsics relative to reference
    """
    ref_4x4 = to_4x4(reference_extrinsic)
    # ref_inv = np.linalg.inv(ref_4x4)
    
    transformed_extrinsics = []
    for i in range(extrinsics_window.shape[0]):
        current_4x4 = to_4x4(extrinsics_window[i])
        transformed_4x4 = current_4x4 @ ref_4x4
        transformed_3x4 = to_3x4(transformed_4x4)
        transformed_extrinsics.append(transformed_3x4)
    
    return np.stack(transformed_extrinsics)


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
    
    # Add final window if necessary to cover all frames
    if window_starts[-1] + window_size < total_frames:
        window_starts.append(total_frames - window_size)
    
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
    
    # Global reference transformation (from first window, first frame)
    global_reference_extrinsic = None
    
    # Store previous window results for overlap-based alignment
    previous_window_extrinsics = None
    previous_window_end_idx = None
    
    # Process each window
    for window_idx, start_idx in enumerate(tqdm(window_starts, desc="Processing windows")):
        end_idx = min(start_idx + window_size, total_frames)
        window_image_paths = image_paths[start_idx:end_idx]
        
        print(f"\nWindow {window_idx + 1}/{len(window_starts)}: frames {start_idx}-{end_idx-1}")
        
        try:
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
            world_points = predictions["world_points"].squeeze(0).cpu().numpy()  # (S, H, W, 3)
            world_points_conf = predictions["world_points_conf"].squeeze(0).cpu().numpy()  # (S, H, W)
            depth_maps = predictions["depth"].squeeze(0).cpu().numpy()     # (S, H, W, 1)
            depth_confs = predictions["depth_conf"].squeeze(0).cpu().numpy() # (S, H, W)
            
            # Handle alignment based on window
            if window_idx == 0:
                previous_window_extrinsics = extrinsic_window
                previous_window_end_idx = end_idx
            if window_idx > 0:
                assert previous_window_extrinsics is not None

                overlap_start_global = max(0, start_idx)
                overlap_end_global = min(start_idx + overlap, previous_window_end_idx)
                actual_overlap = overlap_end_global - overlap_start_global
                prev_overlap_frames = []
                for global_idx in range(overlap_start_global, overlap_end_global):
                    if global_idx in unified_results['extrinsics']:
                        prev_overlap_frames.append(unified_results['extrinsics'][global_idx])
                
                if len(prev_overlap_frames) > 0:
                    prev_overlap_extrinsics = np.stack(prev_overlap_frames)
                    
                    # Get current window's overlapping frames (in local coordinates)
                    curr_overlap_end = len(prev_overlap_frames)
                    curr_overlap_extrinsics = previous_window_extrinsics[overlap_start_global:]
                    
                    transformed_nonoverlap_extrinsics = transform_poses_to_reference(
                        extrinsic_window[curr_overlap_end:], previous_window_extrinsics[overlap_start_global]
                    )

                    transformed_extrinsics = np.concatenate([curr_overlap_extrinsics, transformed_nonoverlap_extrinsics], axis=0)
                    previous_window_extrinsics = transformed_extrinsics
            
            # Store results for each frame in the window
            # FIXED: Don't use offset, just store all frames and let the overlap check handle duplicates
            for i in range(len(transformed_extrinsics)):
                global_frame_idx = start_idx + i
                
                # For overlapping frames, keep the result from the first window that processed them
                if global_frame_idx not in unified_results['extrinsics']:
                    unified_results['extrinsics'][global_frame_idx] = transformed_extrinsics[i]
                    unified_results['intrinsics'][global_frame_idx] = intrinsic_window[i]
                    unified_results['images'][global_frame_idx] = images_np[i]
                    unified_results['world_points'][global_frame_idx] = world_points[i]
                    unified_results['world_points_conf'][global_frame_idx] = world_points_conf[i]
                    unified_results['depth'][global_frame_idx] = depth_maps[i]
                    unified_results['depth_conf'][global_frame_idx] = depth_confs[i]
                else:
                    # For overlapping frames, we keep the previous result
                    print(f"Frame {global_frame_idx} already processed, keeping original result")
            
            # Update previous window data for next iteration
            previous_window_extrinsics = transformed_extrinsics.copy()
            previous_window_end_idx = end_idx
            
            # Clear GPU memory
            del predictions, images
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing window {window_idx}: {e}")
            continue
    
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
    image_paths = new_image_paths[48:57]
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

    # Save results if requested
    if args.save_results:
        print(f"Saving unified results to {args.save_results}")
        np.savez_compressed(args.save_results, **predictions)

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")
    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()
