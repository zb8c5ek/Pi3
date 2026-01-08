import os
import os.path as osp
import math
import cv2
from PIL import Image
import torch
from torchvision import transforms
from plyfile import PlyData, PlyElement
import numpy as np

def load_images_as_tensor(path="data/truck", interval=1, PIXEL_LIMIT=255000, verbose=True):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = []

    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        if verbose:
            print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith((".png", ".jpg", ".jpeg"))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith(".mp4"):
        if verbose:
            print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if verbose:
        print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()

    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


def load_multimodal_data(path="data/truck", conditions=None, interval=1, PIXEL_LIMIT=255000, verbose=True, device='cpu'):
    """
    Loads images (using strict original logic) and aligns optional conditions (poses, depths, intrinsics).
    
    Args:
        path: Path to images or video.
        conditions: Dict containing numpy arrays:
                    - 'intrinsics': (N_total, 3, 3)
                    - 'poses': (N_total, 4, 4)
                    - 'depths': (N_total, H, W)
        interval: Sampling interval.
    
    Returns:
        dict: {
            'images': (N, 3, H, W),
            'poses': (N, 4, 4) or None,
            'depths': (N, H, W) or None,
            'intrinsics': (N, 3, 3) or None
        }
    """
    sources = []

    # --- 1. Load image paths or video frames ---
    if osp.isdir(path):
        if verbose:
            print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith((".png", ".jpg", ".jpeg"))])
        for i in range(0, len(filenames), interval):
            img_path = osp.join(path, filenames[i])
            try:
                sources.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"Could not load image {filenames[i]}: {e}")
    elif path.lower().endswith(".mp4"):
        if verbose:
            print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise OSError(f"Cannot open video file: {path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sources.append(Image.fromarray(rgb_frame))
            frame_idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path. Must be a directory or a .mp4 file: {path}")

    if not sources:
        print("No images found or loaded.")
        return {'images': torch.empty(0)}

    if verbose:
        print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = transforms.ToTensor()

    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return {'images': torch.empty(0)}

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    images_tensor = torch.stack(tensor_list, dim=0)
    
    # =========================================================================
    # NEW: Process Conditions (Poses, Depths, Intrinsics)
    # =========================================================================
    
    N_out = images_tensor.shape[0] # The actual number of successfully loaded frames
    
    out_poses = None
    out_depths = None
    out_intrinsics = None

    if conditions is not None:
        # Calculate resize ratios for geometry alignment
        # (Must be calculated here because TARGET_W/H might have been adjusted by the while loop above)
        scale_x = TARGET_W / W_orig
        scale_y = TARGET_H / H_orig

        # --- A. Process Poses (Slice) ---
        if 'poses' in conditions and conditions['poses'] is not None:
            raw_poses = conditions['poses'] # Expected numpy (N_total, 4, 4)
            # Apply interval and trim to N_out
            sliced_poses = raw_poses[::interval][:N_out]
            out_poses = torch.from_numpy(sliced_poses).float()[None].to(device) # (N, 4, 4)

        # --- B. Process Depths (Slice + Resize) ---
        if 'depths' in conditions and conditions['depths'] is not None:
            raw_depths = conditions['depths'] # Expected numpy (N_total, H_orig, W_orig)
            sliced_depths = raw_depths[::interval][:N_out]
            
            resized_depths_list = []
            for d_map in sliced_depths:
                # Use Nearest Neighbor for depth to avoid flying pixels
                # cv2.resize expects (width, height)
                d_resized = cv2.resize(d_map, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

                valid_depth = np.logical_and(d_resized > 0, np.isfinite(d_resized))
                d_resized[~valid_depth] = 0

                resized_depths_list.append(torch.from_numpy(d_resized))
            
            if resized_depths_list:
                out_depths = torch.stack(resized_depths_list, dim=0)[None].to(device) # (N, H, W)

        # --- C. Process Intrinsics (Slice + Rescale) ---
        if 'intrinsics' in conditions and conditions['intrinsics'] is not None:
            raw_intrinsics = conditions['intrinsics'] # Expected numpy (N_total, 3, 3)
            # Copy to avoid modifying original dict
            sliced_Ks = raw_intrinsics[::interval][:N_out].copy()
            
            # Rescale (fx, fy, cx, cy)
            # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            sliced_Ks[:, 0, 0] *= scale_x # fx
            sliced_Ks[:, 0, 2] *= scale_x # cx
            sliced_Ks[:, 1, 1] *= scale_y # fy
            sliced_Ks[:, 1, 2] *= scale_y # cy
            
            out_intrinsics = torch.from_numpy(sliced_Ks).float()[None].to(device) # (N, 3, 3)

    return images_tensor[None].to(device), {
        'poses': out_poses,            # (N, 4, 4)
        'depths': out_depths,          # (N, H, W)
        'intrinsics': out_intrinsics   # (N, 3, 3)
    }

def tensor_to_pil(tensor):
    """
    Converts a PyTorch tensor to a PIL image. Automatically moves the channel dimension 
    (if it has size 3) to the last axis before converting.

    Args:
        tensor (torch.Tensor): Input tensor. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor

    return array_to_pil(array)


def array_to_pil(array):
    """
    Converts a NumPy array to a PIL image. Automatically:
        - Squeezes dimensions of size 1.
        - Moves the channel dimension (if it has size 3) to the last axis.
    
    Args:
        array (np.ndarray): Input array. Expected shape can be [C, H, W], [H, W, C], or [H, W].
    
    Returns:
        PIL.Image: The converted PIL image.
    """
    # Remove singleton dimensions
    array = np.squeeze(array)
    
    # Ensure the array has the channel dimension as the last axis
    if array.ndim == 3 and array.shape[0] == 3:  # If the channel is the first axis
        array = np.transpose(array, (1, 2, 0))  # Move channel to the last axis
    
    # Handle single-channel grayscale images
    if array.ndim == 2:  # [H, W]
        return Image.fromarray((array * 255).astype(np.uint8), mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:  # [H, W, C] with 3 channels
        return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for PIL conversion: {array.shape}")


def rotate_target_dim_to_last_axis(x, target_dim=3):
    shape = x.shape
    axis_to_move = -1
    # Iterate backwards to find the first occurrence from the end 
    # (which corresponds to the last dimension of size 3 in the original order).
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == target_dim:
            axis_to_move = i
            break

    # 2. If the axis is found and it's not already in the last position, move it.
    if axis_to_move != -1 and axis_to_move != len(shape) - 1:
        # Create the new dimension order.
        dims_order = list(range(len(shape)))
        dims_order.pop(axis_to_move)
        dims_order.append(axis_to_move)
        
        # Use permute to reorder the dimensions.
        ret = x.transpose(*dims_order)
    else:
        ret = x

    return ret


def write_ply(
    xyz,
    rgb=None,
    path='output.ply',
) -> None:
    """Write point cloud to PLY file.
    
    Args:
        xyz: Point positions
        rgb: Point colors
        path: Output file path
    """
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()

    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()

    if rgb is not None and rgb.max() > 1:
        rgb = rgb / 255.

    xyz = rotate_target_dim_to_last_axis(xyz, 3)
    xyz = xyz.reshape(-1, 3)

    if rgb is not None:
        rgb = rotate_target_dim_to_last_axis(rgb, 3)
        rgb = rgb.reshape(-1, 3)
    
    if rgb is None:
        min_coord = np.min(xyz, axis=0)
        max_coord = np.max(xyz, axis=0)
        normalized_coord = (xyz - min_coord) / (max_coord - min_coord + 1e-8)
        
        hue = 0.7 * normalized_coord[:,0] + 0.2 * normalized_coord[:,1] + 0.1 * normalized_coord[:,2]
        hsv = np.stack([hue, 0.9*np.ones_like(hue), 0.8*np.ones_like(hue)], axis=1)

        c = hsv[:,2:] * hsv[:,1:2]
        x = c * (1 - np.abs( (hsv[:,0:1]*6) % 2 - 1 ))
        m = hsv[:,2:] - c
        
        rgb = np.zeros_like(hsv)
        cond = (0 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 1)
        rgb[cond] = np.hstack([c[cond], x[cond], np.zeros_like(x[cond])])
        cond = (1 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 2)
        rgb[cond] = np.hstack([x[cond], c[cond], np.zeros_like(x[cond])])
        cond = (2 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 3)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), c[cond], x[cond]])
        cond = (3 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 4)
        rgb[cond] = np.hstack([np.zeros_like(x[cond]), x[cond], c[cond]])
        cond = (4 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 5)
        rgb[cond] = np.hstack([x[cond], np.zeros_like(x[cond]), c[cond]])
        cond = (5 <= hsv[:,0]*6%6) & (hsv[:,0]*6%6 < 6)
        rgb[cond] = np.hstack([c[cond], np.zeros_like(x[cond]), x[cond]])
        rgb = (rgb + m)

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb * 255), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def write_camera_trajectory(
    camera_poses,
    path='camera_trajectory.npz',
) -> None:
    """
    Save camera trajectory (camera-to-world transformation matrices).
    
    Args:
        camera_poses: Camera-to-world matrices, shape (B, N, 4, 4) or (N, 4, 4)
        path: Path to save the .npz file
    """
    if torch.is_tensor(camera_poses):
        camera_poses = camera_poses.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(camera_poses.shape) == 4:
        if camera_poses.shape[0] == 1:
            camera_poses = camera_poses[0]
    
    # Save as numpy array
    np.savez(path, camera_poses=camera_poses)
    print(f"Camera trajectory saved to: {path}")


def write_camera_trajectory_txt(
    camera_poses,
    path='camera_trajectory.txt',
) -> None:
    """
    Save camera trajectory as a text file (one camera position per line).
    
    Args:
        camera_poses: Camera-to-world matrices, shape (B, N, 4, 4) or (N, 4, 4)
        path: Path to save the .txt file
    """
    if torch.is_tensor(camera_poses):
        camera_poses = camera_poses.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(camera_poses.shape) == 4:
        if camera_poses.shape[0] == 1:
            camera_poses = camera_poses[0]
    
    # Extract camera centers
    camera_centers = camera_poses[:, :3, 3]
    
    with open(path, 'w') as f:
        f.write("# Camera trajectory (camera centers)\n")
        f.write("# x y z\n")
        for i, center in enumerate(camera_centers):
            f.write(f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")
    
    print(f"Camera trajectory (txt) saved to: {path}")


def write_camera_trajectory_json(
    camera_poses,
    path='camera_trajectory.json',
) -> None:
    """
    Save camera trajectory as a JSON file.
    
    Args:
        camera_poses: Camera-to-world matrices, shape (B, N, 4, 4) or (N, 4, 4)
        path: Path to save the .json file
    """
    import json
    
    if torch.is_tensor(camera_poses):
        camera_poses = camera_poses.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(camera_poses.shape) == 4:
        if camera_poses.shape[0] == 1:
            camera_poses = camera_poses[0]
    
    # Extract camera centers
    camera_centers = camera_poses[:, :3, 3]
    
    data = {
        "num_cameras": len(camera_centers),
        "camera_centers": camera_centers.tolist(),
        "camera_poses": camera_poses.tolist(),
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Camera trajectory (json) saved to: {path}")


def write_camera_frustums_ply(
    camera_poses,
    pointcloud_xyz=None,
    path='trajectory.ply',
    frustum_scale=0.1,
) -> None:
    """
    Save camera trajectory with frustum visualization as PLY file.
    
    Creates a square pyramid (五面体) for each camera showing its position and orientation.
    The apex points at the camera origin, and the square base face towards the camera's viewing direction.
    
    Args:
        camera_poses: Camera-to-world matrices, shape (B, N, 4, 4) or (N, 4, 4)
        pointcloud_xyz: Optional point cloud to include in the PLY
        path: Path to save the .ply file
        frustum_scale: Scale of the camera frustum pyramids
    """
    if torch.is_tensor(camera_poses):
        camera_poses = camera_poses.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(camera_poses.shape) == 4:
        if camera_poses.shape[0] == 1:
            camera_poses = camera_poses[0]
    
    num_cameras = len(camera_poses)
    
    # Start with point cloud if provided
    if pointcloud_xyz is not None:
        if torch.is_tensor(pointcloud_xyz):
            pointcloud_xyz = pointcloud_xyz.detach().cpu().numpy()
        all_vertices = [pointcloud_xyz]
        vertex_offset = len(pointcloud_xyz)
    else:
        all_vertices = []
        vertex_offset = 0
    
    # Create square pyramid (五面体) vertices in camera frame
    # Apex at origin, base square at distance along +Z axis (camera viewing direction)
    frustum_vertices_template = np.array([
        [0, 0, 0],  # Apex at camera center
        # Base vertices (square) at viewing direction
        [frustum_scale, frustum_scale, frustum_scale],   # Right-Top
        [-frustum_scale, frustum_scale, frustum_scale],  # Left-Top
        [-frustum_scale, -frustum_scale, frustum_scale], # Left-Bottom
        [frustum_scale, -frustum_scale, frustum_scale],  # Right-Bottom
    ], dtype=np.float32)
    
    # Colors for pyramids (blue to red gradient based on camera index)
    frustum_faces = []
    all_colors = []
    
    for i, pose in enumerate(camera_poses):
        # Transform frustum vertices to world coordinates
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        
        # Apply camera-to-world transformation
        world_vertices = frustum_vertices_template @ rotation.T + translation
        
        # Add to vertices list
        all_vertices.append(world_vertices)
        
        # Create color for this frustum (gradient from blue to red)
        t = i / max(1, num_cameras - 1)
        color = np.array([
            int(t * 255),           # Red
            100,                     # Green (fixed)
            int((1 - t) * 255),     # Blue
        ], dtype=np.uint8)
        
        # Add color for each vertex of this pyramid
        all_colors.extend([color] * 5)
        
        # Define pyramid faces:
        # Face 0: Square base (1, 2, 3, 4) - pointing towards viewing direction
        # Faces 1-4: Triangular side faces connecting apex to base edges
        base_idx = vertex_offset + i * 5
        
        # Square base face
        frustum_faces.append([base_idx + 1, base_idx + 2, base_idx + 3])
        frustum_faces.append([base_idx + 1, base_idx + 3, base_idx + 4])
        
        # Triangular side faces (connecting apex to base edges)
        frustum_faces.append([base_idx + 0, base_idx + 1, base_idx + 2])  # Top side
        frustum_faces.append([base_idx + 0, base_idx + 2, base_idx + 3])  # Left side
        frustum_faces.append([base_idx + 0, base_idx + 3, base_idx + 4])  # Bottom side
        frustum_faces.append([base_idx + 0, base_idx + 4, base_idx + 1])  # Right side
    
    # Combine all vertices
    vertices = np.vstack(all_vertices)
    
    # Handle colors for point cloud (if included)
    if pointcloud_xyz is not None:
        # Reuse point cloud colors (default grayscale)
        pointcloud_colors = np.zeros((len(pointcloud_xyz), 3), dtype=np.uint8)
        # Color by height or use default
        min_z = pointcloud_xyz[:, 2].min()
        max_z = pointcloud_xyz[:, 2].max()
        if max_z > min_z:
            normalized_z = (pointcloud_xyz[:, 2] - min_z) / (max_z - min_z)
            pointcloud_colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # Red channel
            pointcloud_colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel
        else:
            pointcloud_colors[:, 0:3] = 128
        
        all_colors = np.vstack([pointcloud_colors, np.array(all_colors)])
    else:
        all_colors = np.array(all_colors)
    
    # Create PLY elements
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    
    vertex_data = np.empty(len(vertices), dtype=dtype)
    vertex_data["x"] = vertices[:, 0]
    vertex_data["y"] = vertices[:, 1]
    vertex_data["z"] = vertices[:, 2]
    vertex_data["red"] = all_colors[:, 0]
    vertex_data["green"] = all_colors[:, 1]
    vertex_data["blue"] = all_colors[:, 2]
    
    vertex_element = PlyElement.describe(vertex_data, "vertex")
    
    # Create face element
    if frustum_faces:
        face_dtype = [("vertex_indices", "i4", (3,))]
        face_data = np.array(
            [(faces,) for faces in frustum_faces],
            dtype=face_dtype
        )
        face_element = PlyElement.describe(face_data, "face")
        ply_data = PlyData([vertex_element, face_element])
    else:
        ply_data = PlyData([vertex_element])
    
    ply_data.write(path)
    print(f"Camera frustums (trajectory.ply) saved to: {path}")