"""
Convert COLMAP camera poses to Pi3X format.

COLMAP stores poses as world-to-camera (W2C), while Pi3X expects camera-to-world (C2W) in OpenCV convention.
This script reads COLMAP sparse reconstruction and creates a .npz file compatible with Pi3X.

Usage:
    python convert_colmap_to_pi3x.py --colmap_path <path_to_sparse/0> --output_path conditions.npz
"""

import os
import argparse
import numpy as np
from pathlib import Path


def read_colmap_cameras(cameras_file):
    """
    Read COLMAP cameras.txt file.
    Returns dict: camera_id -> (model, width, height, params)
    """
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]])
            cameras[camera_id] = (model, width, height, params)
    return cameras


def read_colmap_images(images_file):
    """
    Read COLMAP images.txt file.
    Returns list of (image_name, qvec, tvec, camera_id)
    """
    images = []
    with open(images_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
    # images.txt has pairs of lines: image info, then points (we only need image info)
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        image_name = parts[9]
        
        images.append({
            'image_id': image_id,
            'image_name': image_name,
            'qvec': np.array([qw, qx, qy, qz]),
            'tvec': np.array([tx, ty, tz]),
            'camera_id': camera_id
        })
    
    # Sort by image_id to maintain order
    images.sort(key=lambda x: x['image_id'])
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def colmap_to_opencv_pose(qvec, tvec):
    """
    Convert COLMAP pose (world-to-camera) to OpenCV camera-to-world format.
    
    COLMAP: W2C (world to camera), with Y-down Z-forward convention
    Pi3X expects: C2W (camera to world), OpenCV convention (Y-down Z-forward)
    
    COLMAP stores: R_w2c and t_w2c such that x_cam = R_w2c * x_world + t_w2c
    We need: 4x4 C2W matrix in OpenCV format
    """
    # Get rotation matrix from quaternion (W2C)
    R_w2c = qvec2rotmat(qvec)
    t_w2c = tvec
    
    # Invert to get C2W
    # For a camera pose: [R | t], the inverse is [R^T | -R^T @ t]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    
    # Build 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = R_c2w
    pose[:3, 3] = t_c2w
    
    return pose


def camera_params_to_intrinsics(model, width, height, params):
    """
    Convert COLMAP camera parameters to 3x3 intrinsic matrix.
    
    Supports: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL
    """
    K = np.eye(3)
    
    if model == 'SIMPLE_PINHOLE':
        # params: f, cx, cy
        f, cx, cy = params
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif model == 'PINHOLE':
        # params: fx, fy, cx, cy
        fx, fy, cx, cy = params
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    elif model in ['SIMPLE_RADIAL', 'RADIAL']:
        # params: f, cx, cy, k1, [k2]
        f, cx, cy = params[:3]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif model == 'OPENCV':
        # params: fx, fy, cx, cy, k1, k2, p1, p2
        fx, fy, cx, cy = params[:4]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    else:
        print(f"Warning: Unsupported camera model '{model}'. Using identity intrinsics.")
    
    return K


def convert_colmap_to_pi3x(colmap_path, output_path, image_dir=None):
    """
    Convert COLMAP sparse reconstruction to Pi3X condition format.
    
    Args:
        colmap_path: Path to COLMAP sparse reconstruction (e.g., sparse/0)
        output_path: Output .npz file path
        image_dir: Optional path to images directory (for validation)
    """
    colmap_path = Path(colmap_path)
    
    # Check if files exist
    cameras_file = colmap_path / 'cameras.txt'
    images_file = colmap_path / 'images.txt'
    
    if not cameras_file.exists():
        cameras_file = colmap_path / 'cameras.bin'
        if not cameras_file.exists():
            raise FileNotFoundError(f"Cannot find cameras file in {colmap_path}")
    
    if not images_file.exists():
        images_file = colmap_path / 'images.bin'
        if not images_file.exists():
            raise FileNotFoundError(f"Cannot find images file in {colmap_path}")
    
    # For .bin files, use a different reader (not implemented here for simplicity)
    if cameras_file.suffix == '.bin':
        raise NotImplementedError(
            "Binary COLMAP files not supported. Please convert to .txt format using:\n"
            "colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT"
        )
    
    print(f"Reading COLMAP data from: {colmap_path}")
    
    # Read COLMAP data
    cameras = read_colmap_cameras(cameras_file)
    images = read_colmap_images(images_file)
    
    print(f"Found {len(cameras)} cameras and {len(images)} images")
    
    # Convert to Pi3X format
    N = len(images)
    poses_list = []
    intrinsics_list = []
    image_names = []
    
    for img_data in images:
        # Get camera info
        camera_id = img_data['camera_id']
        model, width, height, params = cameras[camera_id]
        
        # Convert pose (W2C -> C2W in OpenCV format)
        pose = colmap_to_opencv_pose(img_data['qvec'], img_data['tvec'])
        poses_list.append(pose)
        
        # Convert intrinsics
        K = camera_params_to_intrinsics(model, width, height, params)
        intrinsics_list.append(K)
        
        image_names.append(img_data['image_name'])
    
    # Stack into arrays
    poses = np.stack(poses_list, axis=0)  # (N, 4, 4)
    intrinsics = np.stack(intrinsics_list, axis=0)  # (N, 3, 3)
    
    print(f"\nConverted data:")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Intrinsics shape: {intrinsics.shape}")
    print(f"  Images: {image_names[:5]}{'...' if len(image_names) > 5 else ''}")
    
    # Save to .npz
    np.savez(
        output_path,
        poses=poses,
        intrinsics=intrinsics,
        image_names=np.array(image_names),
        depths=None  # No depth data from COLMAP
    )
    
    print(f"\n✓ Saved conditions to: {output_path}")
    print(f"\nUsage:")
    print(f"  python example_mm.py --data_path <your_images_dir> --conditions_path {output_path} --save_path output.ply")
    
    return poses, intrinsics, image_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert COLMAP poses to Pi3X format')
    parser.add_argument('--colmap_path', type=str, required=True,
                        help='Path to COLMAP sparse reconstruction (e.g., sparse/0)')
    parser.add_argument('--output_path', type=str, default='conditions.npz',
                        help='Output .npz file path')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Optional path to images directory for validation')
    
    args = parser.parse_args()
    
    convert_colmap_to_pi3x(args.colmap_path, args.output_path, args.image_dir)
