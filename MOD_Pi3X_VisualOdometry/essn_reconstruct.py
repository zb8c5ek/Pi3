"""
essn_reconstruct - High-level reconstruction API

Integrates kern_colmap and kern_inference to provide a simple interface
for running Pi3X visual odometry reconstruction.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch
import numpy as np

from pi3.utils.basic import write_ply

from .kern_colmap import write_colmap_txt
from .kern_inference import load_model, load_images_multicam, run_inference, filter_points


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction."""
    # Input
    imgs_root: str = ""
    cam_dirs: List[str] = field(default_factory=lambda: ["cam0", "cam1"])
    interval: int = 1
    
    # Model
    ckpt_path: str = r"D:\_HUBs\HuggingFace\Pi3X\model.safetensors"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Processing
    pixel_limit: int = 255000
    conf_threshold: float = 0.1
    edge_rtol: float = 0.03
    subsample_ratio: float = 0.01
    
    # Output
    output_root: str = ""
    save_ply: bool = True
    save_colmap: bool = True


def reconstruct(
    config: Union[ReconstructionConfig, Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Run full reconstruction pipeline.
    
    Args:
        config: ReconstructionConfig or dict with config values
        output_dir: Override output directory (optional)
        
    Returns:
        Dict with reconstruction results and metadata
    """
    # Convert dict to config if needed
    if isinstance(config, dict):
        config = ReconstructionConfig(**config)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        output_path = Path(output_dir)
    elif config.output_root:
        output_path = Path(config.output_root) / f"reconstruction_{timestamp}"
    else:
        output_path = Path(config.imgs_root).parent / "outputs" / f"reconstruction_{timestamp}"
    
    output_path.mkdir(parents=True, exist_ok=True)
    colmap_path = output_path / "colmap"
    colmap_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)
    
    # Load model
    print("Loading model...")
    model, model_load_time = load_model(config.ckpt_path, config.device)
    print(f"Model loaded in {model_load_time:.2f}s")
    
    # Load images
    print("Loading images...")
    imgs, image_names, (H, W), data_load_time = load_images_multicam(
        config.imgs_root,
        config.cam_dirs,
        interval=config.interval,
        pixel_limit=config.pixel_limit,
    )
    print(f"Data loaded in {data_load_time:.2f}s")
    
    # Run inference
    print("Running model inference...")
    results, inference_time = run_inference(model, imgs, config.device, dtype)
    print(f"Inference completed in {inference_time:.2f}s")
    print("Reconstruction complete!")
    
    # Move to CPU
    print("Moving results to CPU...")
    results_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in results.items()}
    imgs_cpu = imgs.cpu()
    del results, imgs, model
    torch.cuda.empty_cache()
    
    # Filter points
    print("Processing point cloud...")
    points, colors, total_before = filter_points(
        results_cpu,
        imgs_cpu,
        conf_threshold=config.conf_threshold,
        edge_rtol=config.edge_rtol,
        subsample_ratio=config.subsample_ratio,
    )
    print(f"Total points after filtering: {total_before}")
    if config.subsample_ratio < 1.0:
        print(f"Subsampled to {len(points)} points ({config.subsample_ratio*100:.1f}%)")
    
    # Save outputs
    ply_path = None
    if config.save_ply:
        ply_path = output_path / "pointcloud.ply"
        print(f"Saving PLY to: {ply_path}")
        write_ply(points, colors, str(ply_path))
    
    if config.save_colmap:
        print(f"Saving COLMAP format to: {colmap_path}")
        poses_np = results_cpu['camera_poses'][0].numpy()
        points_np = points.numpy()
        colors_np = (colors.numpy() * 255).astype(np.uint8)
        write_colmap_txt(str(colmap_path), poses_np, image_names, points_np, colors_np, H, W)
        print(f"Saved {len(points_np)} points to COLMAP format")
    
    # Build results info
    results_info = {
        "timestamp": timestamp,
        "output_dir": str(output_path),
        "timing": {
            "model_load_time_s": round(model_load_time, 3),
            "data_load_time_s": round(data_load_time, 3),
            "inference_time_s": round(inference_time, 3),
            "total_time_s": round(model_load_time + data_load_time + inference_time, 3),
        },
        "input": {
            "data_path": str(config.imgs_root),
            "cam_dirs": config.cam_dirs,
            "num_images": imgs_cpu.shape[0],
            "image_shape": list(imgs_cpu.shape[1:]),
            "device": config.device,
            "dtype": config.dtype,
        },
        "filtering": {
            "subsample_ratio": config.subsample_ratio,
            "conf_threshold": config.conf_threshold,
            "edge_rtol": config.edge_rtol,
            "points_before_subsample": total_before,
        },
        "output": {
            "ply_path": str(ply_path) if ply_path else None,
            "colmap_dir": str(colmap_path) if config.save_colmap else None,
            "num_points": len(points),
        },
        "model_outputs": {},
    }
    
    for key, value in results_cpu.items():
        if isinstance(value, torch.Tensor):
            results_info["model_outputs"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    
    # Save results info
    json_path = output_path / "results_info.json"
    with open(json_path, 'w') as f:
        json.dump(results_info, f, indent=2)
    print(f"Results info saved to: {json_path}")
    
    print("\n" + "=" * 50)
    print("Results Summary:")
    print("=" * 50)
    print(json.dumps(results_info, indent=2))
    print("\nDone!")
    
    return results_info
