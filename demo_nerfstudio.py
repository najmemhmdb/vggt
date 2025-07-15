import json
import os
import numpy as np
import trimesh
import pycolmap
import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import shutil
import math
import cv2
# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import argparse
from pathlib import Path
import trimesh
import pycolmap
from scipy.spatial.transform import Rotation as R, Slerp
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from nerfstudio.process_data.colmap_utils import colmap_to_json, create_sfm_depth
from pathlib import Path

def visualize_colmap_with_viser(sparse_dir: str, images_dir: str, port: int = 8081):
    """
    Launch a ViserServer to display:
      • the COLMAP point cloud (points3D.bin)
      • camera frusta at each pose (images.bin + cameras.bin)
    Requires: pip install viser[extras]
    """
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(titlebar_content=None)

    # Read COLMAP outputs
    cam_path   = Path(sparse_dir) / "cameras.bin"
    img_path   = Path(sparse_dir) / "images.bin"
    pts3d_path = Path(sparse_dir) / "points3D.bin"
    cameras    = read_cameras_binary(cam_path)
    images     = read_images_binary(img_path)
    points3d   = read_points3d_binary(pts3d_path)

    # Point cloud
    pts  = np.array([p.xyz for p in points3d.values()])
    cols = np.array([p.rgb for p in points3d.values()], dtype=np.uint8)
    server.scene.add_point_cloud(
        "/scene/points",
        points=pts,
        colors=cols,
        point_size=0.01
    )

    # Cameras
    for img in images.values():
        cam = cameras[img.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        w, h = cam.width, cam.height

        # COLMAP qvec = [qw, qx, qy, qz]; Viser wants [x,y,z,w]
        quat_xyzw = np.array([img.qvec[1], img.qvec[2], img.qvec[3], img.qvec[0]])
        trans     = np.array(img.tvec)

       
        fov    = 2 * np.arctan2(h / 2, fy)
        aspect = w / h

        node_name = f"/scene/cameras/{os.path.basename(img.name)}"

        # add just the frustum, positioned & oriented correctly,
        # with no axis triad ever created
        server.scene.add_camera_frustum(
            node_name,
            fov=fov,
            aspect=aspect,
            scale=0.1,
            wxyz=quat_xyzw,
            position=trans,
            color=(255, 0, 0)
        )

    print(f"Viser server running at http://localhost:{port}")
    server.sleep_forever()



def visualize_from_transforms(scene_dir: str,
                              transforms_file: str = "transforms.json",
                              ply_rel_path: str = "sparse/points.ply",
                              port: int = 8080):
    """
    Launch a ViserServer to display:
      • the point cloud from PLY
      • camera frusta from a Nerfstudio‐style transforms.json
    """
    # load transforms.json
    tf_path = Path(scene_dir) / transforms_file
    with open(tf_path, 'r') as f:
        data = json.load(f)

    # pull global intrinsics
    fx = data["fl_x"]
    fy = data["fl_y"]
    W  = data["w"]
    H  = data["h"]

    # start Viser
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(titlebar_content=None)

    # point cloud
    ply_path = Path(scene_dir) / ply_rel_path
    pc = trimesh.load(ply_path)
    pts = np.asarray(pc.vertices)
    cols = np.asarray(pc.visual.vertex_colors[:, :3], dtype=np.uint8)
    server.scene.add_point_cloud(
        "/scene/points",
        points=pts,
        colors=cols,
        point_size=0.01
    )

    # each frame
    for frame in data["frames"]:
        # read the 4×4 transform matrix: camera-to-world
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        R   = mat[:3, :3]
        t   = mat[:3,  3]

        # build quaternion [x,y,z,w] from R
        so3 = vtf.SO3.from_matrix(R)
        quat_xyzw = so3.wxyz  # note: Viser stores wxyz internally, but wxyz attribute gives [w,x,y,z]
        # reorder to [x,y,z,w] for add_camera_frustum
        quat_xyzw = np.array([quat_xyzw[1], quat_xyzw[2], quat_xyzw[3], quat_xyzw[0]])

        # compute vertical FOV and aspect
        fov    = 2 * np.arctan2(H / 2, fy)
        aspect = W / H

        # node name = image filename stem
        img_fp = frame["file_path"]
        name   = Path(img_fp).stem
        node   = f"/scene/transforms/{name}"

        server.scene.add_camera_frustum(
            node,
            fov=fov,
            aspect=aspect,
            scale=0.1,
            wxyz=quat_xyzw,
            position=t,
            color=(0, 255, 0),
        )

    print(f"Viser server running at http://localhost:{port}")
    server.sleep_forever()



def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, default='/mnt/public/Ehsan/datasets/private/Najmeh/real_data/new_lab_v2', help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=True, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument("--freq_factor", type=int, default=3, help="Frequency factor for each round sampling")
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    print('shape of images', images.shape)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        print('shape of aggregated_tokens_list', len(aggregated_tokens_list))
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    print('shape of extrinsic', extrinsic.shape)
    print('shape of intrinsic', intrinsic.shape)
    print('shape of depth_map', depth_map.shape)
    print('shape of depth_conf', depth_conf.shape)
    return extrinsic, intrinsic, depth_map, depth_conf

def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction

def export_depth(output_dir: str, recon_dir: str, image_dir: str):
    depth_dir = output_dir + "/depth"
    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    image_id_to_depth_path = create_sfm_depth(
        recon_dir=recon_dir,
        output_dir=depth_dir,
        input_images_dir=image_dir,
    )
    return image_id_to_depth_path


def export_and_visualize_glb(scene_dir: str,
                             ply_rel_path: str = "sparse/points.ply",
                             glb_rel_path: str = "sparse/scene.glb"):
    """
    Load the reconstructed point cloud (PLY), wrap it in a glTF (GLB), and open an interactive viewer.
    Uses trimesh for both export and display.
    """
    ply_path = os.path.join(scene_dir, ply_rel_path)
    glb_path = os.path.join(scene_dir, glb_rel_path)

    # Load existing PLY point cloud
    pc = trimesh.load(ply_path)
    scene = trimesh.Scene([pc])
    # Export to GLB
    scene.export(glb_path)
    print(f"Exported scene.glb to {glb_path}")
    # Launch trimesh’s built‐in viewer


def interpolate_extrinsic(extrinsic, image_path_list_all, vggt_indices):
    extrinsic_all = np.zeros((len(image_path_list_all), 3, 4), dtype=np.float32)
    for i, idx in enumerate(vggt_indices):
        extrinsic_all[idx] = extrinsic[i]

    # Step 1: Extract rotations and translations
    rotations = R.from_matrix([extrinsic_all[i][:3, :3] for i in vggt_indices])
    translations = np.array([extrinsic_all[i][:3, 3] for i in vggt_indices])

    # Step 2: Interpolation times
    key_times = np.array(vggt_indices)
    interp_times = np.arange(len(image_path_list_all))

    # Step 3: Create interpolators
    slerp = Slerp(key_times, rotations)
    interp_rots = slerp(interp_times)

    interp_trans = np.empty((len(image_path_list_all), 3))
    for i in range(3):  # interpolate each translation axis
        interp_trans[:, i] = np.interp(interp_times, key_times, translations[:, i])

    # Step 4: Compose interpolated extrinsics
    for i in range(len(image_path_list_all)):
        if i in vggt_indices:
            continue
        extrinsic_all[i][:3, :3] = interp_rots[i].as_matrix()
        extrinsic_all[i][:3, 3] = interp_trans[i]
        
    return extrinsic_all


def run(image_path_list_all, img_load_resolution, vggt_fixed_resolution, model, device, dtype, freq_factor):

    # initialize all the arrays
    images_all = np.zeros((len(image_path_list_all), 3, img_load_resolution, img_load_resolution), dtype=np.float32)
    original_coords_all = np.zeros((len(image_path_list_all), 6), dtype=np.float32)
    intrinsic_all = np.zeros((len(image_path_list_all), 3, 3), dtype=np.float32)
    depth_map_all = np.zeros((len(image_path_list_all), vggt_fixed_resolution, vggt_fixed_resolution, 1), dtype=np.float32)
    depth_conf_all = np.zeros((len(image_path_list_all), vggt_fixed_resolution, vggt_fixed_resolution), dtype=np.float32)

    for i in range(freq_factor):
        # Get initial indices for this round
        indices = list(range(i, len(image_path_list_all), freq_factor))
        if i == 0:
            if indices[-1] != len(image_path_list_all) - 1:
                indices.append(len(image_path_list_all) - 1)
            first_round_indices = indices

        else:
            # Avoid duplicate processing of the last item
            if indices and indices[-1] == len(image_path_list_all) - 1 and indices[-1] in first_round_indices:
                indices = indices[:-1]

        # Get the actual image paths for this round
        image_path_list = [image_path_list_all[idx] for idx in indices]
        # Load and preprocess images
        images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
        images = images.to(device)

        # Run VGGT
        extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)

        # First round: interpolate extrinsic
        if i == 0:
            extrinsic_all = interpolate_extrinsic(extrinsic, image_path_list_all, first_round_indices)

        # Save outputs to their proper indices
        for j, idx in enumerate(indices):
            images_all[idx] = images[j].cpu()
            original_coords_all[idx] = original_coords[j]
            intrinsic_all[idx] = intrinsic[j]
            depth_map_all[idx] = depth_map[j]
            depth_conf_all[idx] = depth_conf[j]

    return depth_map_all, depth_conf_all, intrinsic_all, extrinsic_all, images_all, original_coords_all

def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list_all = glob.glob(os.path.join(image_dir, "*"))
    image_path_list_all = sorted(image_path_list_all)
    image_path_list = image_path_list_all[::args.freq_factor]
    if image_path_list[-1] != image_path_list_all[-1]:
        image_path_list.append(image_path_list_all[-1])
    print(f'number of images in the first round: {len(image_path_list)}, total images in the dataset: {len(image_path_list_all)}')

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list_all]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    all_data = run(image_path_list_all, img_load_resolution, vggt_fixed_resolution, model, device, dtype, args.freq_factor)
    depth_map, depth_conf, intrinsic, extrinsic, images, original_coords = all_data
    images = torch.from_numpy(images).to(device)
    original_coords = torch.from_numpy(original_coords).to(device)
    print('shape of images', images.shape)
    print('shape of original_coords', original_coords.shape)
    print('shape of depth_map', depth_map.shape)
    print('shape of depth_conf', depth_conf.shape)
    print('shape of intrinsic', intrinsic.shape)
    print('shape of extrinsic', extrinsic.shape)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    print('shape of points_3d', points_3d.shape)
    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = 2  # hard-coded to 5
        max_points_for_colmap = 10000000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        print('shape of points_xyf', points_xyf.shape)
        conf_mask = depth_conf >= conf_thres_value
        print('shape of conf_mask', conf_mask.shape)
        print('mask sum', conf_mask.sum())
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        print('shape of points_3d', points_3d.shape)
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))


    # Write Nerfstudio JSON
    # save depth maps
    # depth_mm = (depth_map * 1000).astype(np.uint16) 
    # depth_output_dir = Path(args.scene_dir) / "depth"
    # depth_output_dir.mkdir(parents=True, exist_ok=True)
    # image_id_to_depth_path = {}
    # for i, (depth_img, img_path) in enumerate(zip(depth_mm, image_path_list)):
    #     img_name = Path(img_path).name
    #     out_name = img_name.replace(".jpg", ".png").replace(".JPG", ".png")  # keep consistent with transforms.json
    #     depth_path = depth_output_dir / out_name
    #     # Save as 16-bit PNG
    #     cv2.imwrite(str(depth_path), depth_img)
    #     # Record for transforms.json
    #     image_id_to_depth_path[i+1] = depth_path
    image_id_to_depth_path = export_depth(args.scene_dir, Path(os.path.join(args.scene_dir, "sparse")), image_dir)
    colmap_to_json(Path(os.path.join(args.scene_dir, "sparse")), Path(args.scene_dir),None, image_id_to_depth_path, None, "sparse/points.ply", False, False)
    
    # Bundle & view as GLB
    export_and_visualize_glb(args.scene_dir)

    return True

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


    # all_frames = os.listdir("/mnt/public/Ehsan/datasets/private/Najmeh/real_data/new_lab/all_frames")
    # all_frames = sorted(all_frames)
    # print(len(all_frames))
    # idx = list(range(0, len(all_frames), 9))
    # os.makedirs("/mnt/public/Ehsan/datasets/private/Najmeh/real_data/new_lab_v2/images", exist_ok=True)
    # for i in idx:
    #     shutil.copy(os.path.join("/mnt/public/Ehsan/datasets/private/Najmeh/real_data/new_lab/all_frames", all_frames[i]),
    #                  os.path.join("/mnt/public/Ehsan/datasets/private/Najmeh/real_data/new_lab_v2/images", all_frames[i]))
   