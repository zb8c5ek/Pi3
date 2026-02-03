"""
MOD_Pi3X_VisualOdometry - Pi3X Visual Odometry Module

Structure:
- kern_colmap: COLMAP format I/O utilities
- kern_inference: Model loading and inference
- essn_reconstruct: High-level reconstruction API

Usage:
    from MOD_Pi3X_VisualOdometry import reconstruct, ReconstructionConfig
    
    config = ReconstructionConfig(
        imgs_root="path/to/images",
        cam_dirs=["cam0", "cam1"],
        subsample_ratio=0.01,
    )
    results = reconstruct(config)
"""

from .kern_colmap import (
    rotation_matrix_to_quaternion,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
    write_colmap_txt,
)
from .kern_inference import (
    load_model,
    run_inference,
    load_images_multicam,
    filter_points,
)
from .essn_reconstruct import (
    reconstruct,
    ReconstructionConfig,
)

__all__ = [
    # kern_colmap
    'rotation_matrix_to_quaternion',
    'write_cameras_txt',
    'write_images_txt',
    'write_points3d_txt',
    'write_colmap_txt',
    # kern_inference
    'load_model',
    'run_inference',
    'load_images_multicam',
    'filter_points',
    # essn_reconstruct
    'reconstruct',
    'ReconstructionConfig',
]
