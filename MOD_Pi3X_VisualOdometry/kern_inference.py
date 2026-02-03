"""
kern_inference - Model loading and inference utilities

Functions for:
- Loading Pi3X model from checkpoint
- Running inference on image tensors
- Loading images from multi-camera directories
"""

import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from pi3.models.pi3x import Pi3X
from pi3.utils.geometry import depth_edge


def load_model(
    ckpt_path: Union[str, Path],
    device: str = 'cuda',
) -> Tuple[Pi3X, float]:
    """
    Load Pi3X model from checkpoint.
    
    Args:
        ckpt_path: Path to model.safetensors checkpoint
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, load_time_seconds)
    """
    from safetensors.torch import load_file
    
    start_time = time.time()
    model = Pi3X().to(device).eval()
    model.load_state_dict(load_file(ckpt_path), strict=False)
    load_time = time.time() - start_time
    
    return model, load_time


def load_images_multicam(
    imgs_root: Union[str, Path],
    cam_dirs: List[str],
    interval: int = 1,
    pixel_limit: int = 255000,
    patch_size: int = 14,
) -> Tuple[torch.Tensor, List[str], Tuple[int, int], float]:
    """
    Load and preprocess images from multiple camera directories with interleaving.
    
    Images are interleaved across cameras to maintain temporal order:
    cam0[0], cam1[0], cam0[1], cam1[1], ...
    
    Args:
        imgs_root: Root directory containing camera subdirectories
        cam_dirs: List of camera subdirectory names (e.g., ["cam0", "cam1"])
        interval: Frame interval for subsampling
        pixel_limit: Maximum pixels per image (for resizing)
        patch_size: Model patch size (for dimension alignment)
        
    Returns:
        Tuple of (images_tensor, image_names, (H, W), load_time_seconds)
        - images_tensor: (N, 3, H, W) tensor in [0, 1]
        - image_names: List of "cam_dir/filename" strings
        - (H, W): Target image dimensions
        - load_time: Loading time in seconds
    """
    imgs_root = Path(imgs_root)
    start_time = time.time()
    
    # Collect image files from all camera directories
    cam_file_lists = {}
    for cam_dir in cam_dirs:
        cam_path = imgs_root / cam_dir
        if not cam_path.exists():
            print(f"Warning: Camera directory not found: {cam_path}")
            continue
        
        cam_files = sorted([
            f for f in cam_path.glob("*") 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        cam_files = cam_files[::interval]
        cam_file_lists[cam_dir] = cam_files
        print(f"Found {len(cam_files)} images in {cam_dir}/")
    
    if not cam_file_lists:
        raise ValueError("No images found in any camera directory!")
    
    # Interleave images from all cameras
    all_images = []
    image_names = []
    
    max_len = max(len(files) for files in cam_file_lists.values())
    for i in range(max_len):
        for cam_dir in cam_dirs:
            if cam_dir in cam_file_lists and i < len(cam_file_lists[cam_dir]):
                img_file = cam_file_lists[cam_dir][i]
                img = Image.open(img_file).convert("RGB")
                all_images.append(img)
                image_names.append(f"{cam_dir}/{img_file.name}")
    
    print(f"Total images loaded: {len(all_images)} (interleaved from {len(cam_file_lists)} cameras)")
    
    # Calculate target size
    first_img = all_images[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target = int(round(W_orig * scale / patch_size) * patch_size)
    H_target = int(round(H_orig * scale / patch_size) * patch_size)
    print(f"Resizing images to: ({W_target}, {H_target})")
    
    # Transform and stack
    transform = transforms.Compose([
        transforms.Resize((H_target, W_target)),
        transforms.ToTensor(),
    ])
    
    img_tensors = [transform(img) for img in all_images]
    imgs = torch.stack(img_tensors)
    
    load_time = time.time() - start_time
    return imgs, image_names, (H_target, W_target), load_time


def run_inference(
    model: Pi3X,
    imgs: torch.Tensor,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Run Pi3X inference on images.
    
    Args:
        model: Loaded Pi3X model
        imgs: Image tensor (N, 3, H, W) in [0, 1]
        device: Device for inference
        dtype: Data type for mixed precision (float16 recommended)
        
    Returns:
        Tuple of (results_dict, inference_time_seconds)
        Results dict contains:
        - 'points': (1, N, H, W, 3) world coordinates
        - 'local_points': (1, N, H, W, 3) camera-local coordinates
        - 'conf': (1, N, H, W, 1) confidence logits
        - 'camera_poses': (1, N, 4, 4) cam2world poses
        - 'rays': (1, N, H, W, 3) ray directions
        - 'metric': (1,) scale factor
    """
    imgs = imgs.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            results = model(imgs[None])  # Add batch dimension
    torch.cuda.synchronize()
    inference_time = time.time() - start_time
    
    return results, inference_time


def filter_points(
    results: Dict[str, torch.Tensor],
    imgs: torch.Tensor,
    conf_threshold: float = 0.1,
    edge_rtol: float = 0.03,
    subsample_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Filter and subsample point cloud from inference results.
    
    Args:
        results: Inference results dict (on CPU)
        imgs: Image tensor (N, 3, H, W) in [0, 1] (on CPU)
        conf_threshold: Minimum confidence threshold (sigmoid)
        edge_rtol: Edge detection relative tolerance
        subsample_ratio: Fraction of points to keep (1.0 = all)
        
    Returns:
        Tuple of (points, colors, total_before_subsample)
        - points: (M, 3) filtered 3D points
        - colors: (M, 3) RGB colors in [0, 1]
        - total_before_subsample: Number of points before random subsampling
    """
    # Build mask from confidence and edge filter
    conf_scores = torch.sigmoid(results['conf'][..., 0])  # (1, N, H, W)
    masks = conf_scores > conf_threshold
    non_edge = ~depth_edge(results['local_points'][..., 2], rtol=edge_rtol)
    masks = torch.logical_and(masks, non_edge)[0]  # (N, H, W)
    
    # Extract points and colors
    points_all = results['points'][0]  # (N, H, W, 3)
    colors_all = imgs.permute(0, 2, 3, 1)  # (N, H, W, 3)
    
    points_masked = points_all[masks]
    colors_masked = colors_all[masks]
    
    total_before = len(points_masked)
    
    # Random subsampling
    if subsample_ratio < 1.0 and len(points_masked) > 0:
        n_keep = int(len(points_masked) * subsample_ratio)
        indices = torch.randperm(len(points_masked))[:n_keep]
        points_masked = points_masked[indices]
        colors_masked = colors_masked[indices]
    
    return points_masked, colors_masked, total_before
