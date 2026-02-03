"""
kern_colmap - COLMAP format I/O utilities

Functions for reading/writing COLMAP text format files:
- cameras.txt
- images.txt  
- points3D.txt
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion (qw, qx, qy, qz).
    
    Args:
        R: Rotation matrix (3, 3)
        
    Returns:
        List [qw, qx, qy, qz]
    """
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # returns [qx, qy, qz, qw]
    return [float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])]  # [qw, qx, qy, qz]


def write_cameras_txt(output_path, num_images, height, width, focal_length=None):
    """
    Write cameras.txt in COLMAP text format.
    
    Args:
        output_path: Directory to write cameras.txt
        num_images: Number of cameras/images
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Focal length in pixels (estimated if None)
    """
    if focal_length is None:
        focal_length = max(height, width) * 1.2  # rough estimate
    
    cx, cy = width / 2.0, height / 2.0
    
    filepath = os.path.join(output_path, 'cameras.txt')
    with open(filepath, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {num_images}\n")
        # Use PINHOLE model: fx, fy, cx, cy
        for i in range(num_images):
            f.write(f"{i} PINHOLE {width} {height} {focal_length} {focal_length} {cx} {cy}\n")


def write_images_txt(output_path, poses, image_names):
    """
    Write images.txt in COLMAP text format.
    
    Args:
        output_path: Directory to write images.txt
        poses: Camera poses (N, 4, 4) - cam2world matrices
        image_names: List of image filenames
    """
    filepath = os.path.join(output_path, 'images.txt')
    with open(filepath, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}, mean observations per image: 0\n")
        
        for i, (pose, name) in enumerate(zip(poses, image_names)):
            # pose is cam2world (4x4), COLMAP uses world2cam
            # world2cam = inv(cam2world)
            R_c2w = pose[:3, :3]
            t_c2w = pose[:3, 3]
            
            # Inverse: R_w2c = R_c2w.T, t_w2c = -R_c2w.T @ t_c2w
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_w2c)
            tx, ty, tz = t_w2c
            
            # Line 1: Image data
            f.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {name}\n")
            # Line 2: Empty POINTS2D
            f.write("\n")


def write_points3d_txt(output_path, points, colors=None):
    """
    Write points3D.txt in COLMAP text format.
    
    Args:
        output_path: Directory to write points3D.txt
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) in [0, 255], default gray if None
    """
    filepath = os.path.join(output_path, 'points3D.txt')
    with open(filepath, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        
        for i, pt in enumerate(points):
            x, y, z = pt
            if colors is not None:
                r, g, b = colors[i]
            else:
                r, g, b = 128, 128, 128  # default gray
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} -1\n")


def write_colmap_txt(output_path, poses, image_names, points, colors, height, width, focal_length=None):
    """
    Write complete COLMAP text format output.
    
    Args:
        output_path: Directory to write COLMAP files
        poses: Camera poses (N, 4, 4) - cam2world matrices
        image_names: List of image filenames
        points: 3D points (N, 3)
        colors: RGB colors (N, 3) in [0, 255]
        height: Image height
        width: Image width
        focal_length: Focal length (estimated if None)
    """
    os.makedirs(output_path, exist_ok=True)
    write_cameras_txt(output_path, len(poses), height, width, focal_length)
    write_images_txt(output_path, poses, image_names)
    write_points3d_txt(output_path, points, colors)
