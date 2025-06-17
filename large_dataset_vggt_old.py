# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
                # First window: set global reference
                global_reference_extrinsic = extrinsic_window[0].copy()
                print("Set global reference frame from first window")
                
                # Transform all poses in first window relative to the reference
                transformed_extrinsics = transform_poses_to_reference(
                    extrinsic_window, global_reference_extrinsic
                )
                
            else:
                # Subsequent windows: use overlap-based alignment
                if overlap > 0 and previous_window_extrinsics is not None:
                    print(f"Computing overlap-based alignment with {overlap} overlapping frames")
                    
                    # Get overlapping frames
                    overlap_start_global = max(0, start_idx)
                    overlap_end_global = min(start_idx + overlap, previous_window_end_idx)
                    actual_overlap = overlap_end_global - overlap_start_global
                    
                    if actual_overlap > 0:
                        # Get previous window's overlapping frames (already in global coordinates)
                        prev_overlap_frames = []
                        for global_idx in range(overlap_start_global, overlap_end_global):
                            if global_idx in unified_results['extrinsics']:
                                prev_overlap_frames.append(unified_results['extrinsics'][global_idx])
                        
                        if len(prev_overlap_frames) > 0:
                            prev_overlap_extrinsics = np.stack(prev_overlap_frames)
                            
                            # Get current window's overlapping frames (in local coordinates)
                            curr_overlap_start = overlap_start_global - start_idx
                            curr_overlap_end = curr_overlap_start + len(prev_overlap_frames)
                            curr_overlap_extrinsics = extrinsic_window[curr_overlap_start:curr_overlap_end]
                            
                            # Compute alignment transformation
                            alignment_transform = compute_overlap_alignment(
                                prev_overlap_extrinsics,  # Previous window overlap (global coords)
                                curr_overlap_extrinsics,  # Current window overlap (local coords)
                                )
                            
                            # Apply alignment ONLY to non-overlapping frames
                            overlap_count = len(prev_overlap_frames)
                            
                            # Keep overlap frames as-is (they should already be in correct coordinates)
                            overlap_extrinsics = extrinsic_window[:overlap_count]
                            
                            # Transform only the non-overlap frames
                            non_overlap_extrinsics = extrinsic_window[overlap_count:]
                            transformed_non_overlap = apply_alignment_transform(
                                non_overlap_extrinsics, alignment_transform
                            )
                            
                            # Combine: keep overlap frames unchanged, use transformed non-overlap frames
                            transformed_extrinsics = np.concatenate([
                                overlap_extrinsics,      # Keep overlap frames as-is
                                transformed_non_overlap  # Use transformed non-overlap frames
                            ], axis=0)
                            
                            print(f"Applied overlap-based alignment to {len(non_overlap_extrinsics)} non-overlapping frames only")
                        else:
                            print("No valid overlapping frames found, using direct transformation")
                            transformed_extrinsics = transform_poses_to_reference(
                                extrinsic_window, global_reference_extrinsic
                            )
                    else:
                        print("No overlap available, using direct transformation")
                        transformed_extrinsics = transform_poses_to_reference(
                            extrinsic_window, global_reference_extrinsic
                        )
                else:
                    print("No previous window data available, using direct transformation")
                    
                    transformed_extrinsics = transform_poses_to_reference(
                        extrinsic_window, global_reference_extrinsic
                    )
                    
            
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


def compute_overlap_alignment(prev_overlap_extrinsics, curr_overlap_extrinsics):
    """
    Compute alignment transformation between overlapping frames from consecutive windows.
    FIXED VERSION that properly uses extrinsic_to_pose() function with closed_form_inverse_se3.
    
    Args:
        prev_overlap_extrinsics (np.ndarray): Previous window's overlapping frames in global coordinates (N, 3, 4)
        curr_overlap_extrinsics (np.ndarray): Current window's overlapping frames in local coordinates (N, 3, 4)
    
    Returns:
        np.ndarray: Alignment transformation matrix (4, 4)
    """
    assert prev_overlap_extrinsics.shape[0] == curr_overlap_extrinsics.shape[0], \
        "Overlapping frame counts must match"
    
    n_overlap = prev_overlap_extrinsics.shape[0]
    print(f"Computing FIXED alignment using {n_overlap} overlapping frames")
    
    if n_overlap == 1:
        # Simple case: direct transformation between single frames
        # Use extrinsic_to_pose to get proper camera poses with closed_form_inverse_se3
        prev_pose = extrinsic_to_pose(prev_overlap_extrinsics[0])  # Target pose (world-from-camera)
        curr_pose = extrinsic_to_pose(curr_overlap_extrinsics[0])  # Source pose (world-from-camera)
        
        # Compute pose transformation: new_pose = pose_transform @ old_pose
        # So: pose_transform = prev_pose @ inv(curr_pose)
        pose_transform = prev_pose @ np.linalg.inv(curr_pose)
        
        # Convert to extrinsic transformation: alignment_transform = inv(pose_transform)
        alignment_transform = np.linalg.inv(pose_transform)
        
    else:
        # Multiple frames: use least squares alignment with proper extrinsic_to_pose usage
        alignment_transform = compute_least_squares_alignment(
            prev_overlap_extrinsics, curr_overlap_extrinsics
        )
    
    return alignment_transform


def compute_least_squares_alignment(target_extrinsics, source_extrinsics):
    """
    Compute the best-fit rigid transformation that aligns source extrinsics to target extrinsics.
    FIXED to properly use extrinsic_to_pose() function with closed_form_inverse_se3.
    
    Args:
        target_extrinsics (np.ndarray): Target camera extrinsics (N, 3, 4) 
        source_extrinsics (np.ndarray): Source camera extrinsics (N, 3, 4)
    
    Returns:
        np.ndarray: Transformation matrix (4, 4) such that T @ source ≈ target
    """
    n_poses = target_extrinsics.shape[0]
    print(f"Computing FIXED least squares alignment for {n_poses} pose pairs")
    
    # Convert extrinsics to camera poses using the proper function
    target_poses = []
    source_poses = []
    
    for i in range(n_poses):
        # Use the extrinsic_to_pose function that properly handles the conversion with closed_form_inverse_se3
        target_pose = extrinsic_to_pose(target_extrinsics[i])  # world-from-camera
        source_pose = extrinsic_to_pose(source_extrinsics[i])  # world-from-camera
        
        target_poses.append(target_pose)
        source_poses.append(source_pose)
    
    target_poses = np.stack(target_poses)  # (N, 4, 4)
    source_poses = np.stack(source_poses)  # (N, 4, 4)
    
    # Extract camera positions in world coordinates
    target_positions = target_poses[:, :3, 3]  # (N, 3)
    source_positions = source_poses[:, :3, 3]  # (N, 3)
    
    # Compute centroids
    target_centroid = np.mean(target_positions, axis=0)  # (3,)
    source_centroid = np.mean(source_positions, axis=0)  # (3,)
    
    # Center the positions
    target_centered = target_positions - target_centroid  # (N, 3)
    source_centered = source_positions - source_centroid  # (N, 3)
    
    # Compute rotation using Kabsch algorithm
    # We want R such that R @ source_centered ≈ target_centered
    H = source_centered.T @ target_centered  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_centroid - R @ source_centroid
    
    # Build 4x4 transformation matrix for camera poses
    pose_alignment = np.eye(4)
    pose_alignment[:3, :3] = R
    pose_alignment[:3, 3] = t
    
    # Convert back to extrinsic transformation
    # If pose_alignment transforms camera poses: new_pose = pose_alignment @ old_pose
    # Then for extrinsics: new_extrinsic = inv(new_pose) = inv(pose_alignment @ old_pose)
    #                      = inv(old_pose) @ inv(pose_alignment) = old_extrinsic @ inv(pose_alignment)
    # So alignment transform for extrinsics is: inv(pose_alignment)
    alignment_transform = np.linalg.inv(pose_alignment)
    
    # Verify alignment quality
    alignment_error = compute_alignment_error(target_extrinsics, source_extrinsics, alignment_transform)
    print(f"FIXED Alignment error (average translation): {alignment_error:.4f}")
    
    return alignment_transform


def compute_alignment_error(target_extrinsics, source_extrinsics, alignment_transform):
    """
    Compute alignment error between target and transformed source extrinsics.
    FIXED VERSION that uses extrinsic_to_pose() function with closed_form_inverse_se3.
    """
    aligned_extrinsics = apply_alignment_transform(source_extrinsics, alignment_transform)
    
    # Convert to camera positions using proper extrinsic_to_pose function
    target_positions = []
    aligned_positions = []
    
    for i in range(target_extrinsics.shape[0]):
        target_pose = extrinsic_to_pose(target_extrinsics[i])
        aligned_pose = extrinsic_to_pose(aligned_extrinsics[i])
        
        target_positions.append(target_pose[:3, 3])
        aligned_positions.append(aligned_pose[:3, 3])
    
    target_positions = np.array(target_positions)
    aligned_positions = np.array(aligned_positions)
    
    translation_errors = np.linalg.norm(target_positions - aligned_positions, axis=1)
    average_error = np.mean(translation_errors)
    
    return average_error


def apply_alignment_transform(extrinsics, alignment_transform):
    """
    Apply alignment transformation to a set of extrinsic matrices.
    CORRECTED VERSION that properly handles the transformation order.
    
    Args:
        extrinsics (np.ndarray): Extrinsic matrices (N, 3, 4) - camera-from-world
        alignment_transform (np.ndarray): Alignment transformation (4, 4)
    
    Returns:
        np.ndarray: Transformed extrinsics (N, 3, 4)
    """
    n_poses = extrinsics.shape[0]
    transformed_extrinsics = []
    
    for i in range(n_poses):
        # Convert to 4x4
        extrinsic_4x4 = to_4x4(extrinsics[i])  # camera-from-world
        
        # Apply alignment transformation
        # The alignment transform is designed to work on camera-from-world matrices
        aligned_4x4 = alignment_transform @ extrinsic_4x4
        
        # Convert back to 3x4
        aligned_3x4 = to_3x4(aligned_4x4)
        transformed_extrinsics.append(aligned_3x4)
    
    return np.stack(transformed_extrinsics)


def transform_poses_to_reference(extrinsics_window, reference_extrinsic):
    """
    Transform poses in a window to be relative to the reference coordinate system.
    
    Args:
        extrinsics_window (np.ndarray): Window extrinsics of shape (S, 3, 4)
        reference_extrinsic (np.ndarray): Reference extrinsic of shape (3, 4)
    
    Returns:
        np.ndarray: Transformed extrinsics relative to reference
    """
    # Convert reference to 4x4 and get its inverse
    ref_4x4 = to_4x4(reference_extrinsic)
    ref_inv = np.linalg.inv(ref_4x4)
    
    transformed_extrinsics = []
    
    for i in range(extrinsics_window.shape[0]):
        # Convert current extrinsic to 4x4
        current_4x4 = to_4x4(extrinsics_window[i])
        
        # Transform: T_ref_inv * T_current
        # This gives us the pose of current camera relative to reference camera
        transformed_4x4 = ref_inv @ current_4x4
        
        # Convert back to 3x4
        transformed_3x4 = to_3x4(transformed_4x4)
        transformed_extrinsics.append(transformed_3x4)
    
    return np.stack(transformed_extrinsics)


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


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.
    Modified to handle large datasets processed in windows with overlap-based alignment.

    Args:
        pred_dict (dict): Dictionary containing unified predictions from all windows
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")
    print("🔧 USING CORRECTED ALIGNMENT FUNCTION!")  # Debug print
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Confidence threshold slider
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    # Frame selector (with subsampling for large datasets)
    frame_options = ["All"]
    if S > 100:  # If we have many frames, subsample for the dropdown
        step = max(1, S // 50)  # Show at most 50 frame options
        frame_options.extend([str(i) for i in range(0, S, step)])
    else:
        frame_options.extend([str(i) for i in range(S)])

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=frame_options, initial_value="All"
    )

    # Add frame range selector for large datasets
    gui_frame_range_start = server.gui.add_slider(
        "Frame Range Start", min=0, max=S-1, step=1, initial_value=0
    )
    gui_frame_range_end = server.gui.add_slider(
        "Frame Range End", min=0, max=S-1, step=1, initial_value=min(S-1, 100)
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    
    # For large datasets, initially show only a subset of frames
    if S > 100:
        initial_frame_mask = frame_indices < 100
        init_combined_mask = init_conf_mask & initial_frame_mask
    else:
        init_combined_mask = init_conf_mask

    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_combined_mask],
        colors=colors_flat[init_combined_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray, subsample_step: int = 1) -> None:
        """
        Add camera frames and frustums to the scene.
        For large datasets, subsample cameras for visualization.
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        # Subsample frames for visualization if dataset is large
        img_ids = range(0, len(extrinsics), subsample_step)
        print(f"Visualizing {len(list(img_ids))} cameras (every {subsample_step} camera)")
        for img_id in tqdm(img_ids, desc="Adding camera visualizations"):
            # FIX: Convert extrinsic to camera pose properly
            extrinsic_3x4 = extrinsics[img_id]  # This is camera-from-world (3,4)
            cam_to_world_4x4 = extrinsic_to_pose(extrinsic_3x4)  # Convert to world-from-camera (4,4)
            cam_to_world_3x4 = cam_to_world_4x4[:3, :]  # Convert to (3,4)
            
            T_world_camera = viser_tf.SE3.from_matrix(cam_to_world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # Calculate FOV from intrinsics
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Confidence threshold
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        # Frame selection
        if gui_frame_selector.value == "All":
            # Use frame range
            start_frame = gui_frame_range_start.value
            end_frame = gui_frame_range_end.value
            frame_mask = (frame_indices >= start_frame) & (frame_indices <= end_frame)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        
        # Limit number of points for performance
        max_points = 1000000  # 1M points max
        if np.sum(combined_mask) > max_points:
            # Randomly subsample points
            valid_indices = np.where(combined_mask)[0]
            selected_indices = np.random.choice(valid_indices, max_points, replace=False)
            combined_mask = np.zeros_like(combined_mask, dtype=bool)
            combined_mask[selected_indices] = True

        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]
        
        print(f"Displaying {np.sum(combined_mask)} points (threshold: {threshold_val:.3f})")

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_range_start.on_update
    def _(_) -> None:
        if gui_frame_selector.value == "All":
            update_point_cloud()

    @gui_frame_range_end.on_update
    def _(_) -> None:
        if gui_frame_selector.value == "All":
            update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene (subsample for large datasets)
    camera_subsample_step = max(1, S // 50) if S > 50 else 1
    print(f"Visualizing every {camera_subsample_step} camera(s) for {S} total frames")
    visualize_frames(extrinsics_cam, images, camera_subsample_step)

    print("Starting viser server...")
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    sky_mask_array = np.array(sky_mask_list)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


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
