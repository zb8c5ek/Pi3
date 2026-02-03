"""
Example script using MOD_Pi3X_VisualOdometry module.

This demonstrates the high-level API for running Pi3X reconstruction.

from MOD_Pi3X_VisualOdometry import reconstruct, ReconstructionConfig

config = ReconstructionConfig(
    imgs_root="path/to/images",
    cam_dirs=["cam0", "cam1"],
    subsample_ratio=0.01,
)
results = reconstruct(config)

"""

from MOD_Pi3X_VisualOdometry import reconstruct, ReconstructionConfig

# Configure reconstruction
config = ReconstructionConfig(
    # Input
    imgs_root=r"D:\Pi3\assets\cam_0+1_front",
    cam_dirs=["cam0", "cam1"],
    interval=1,
    
    # Model
    ckpt_path=r"D:\_HUBs\HuggingFace\Pi3X\model.safetensors",
    device="cuda",
    dtype="float16",
    
    # Processing
    pixel_limit=255000,
    conf_threshold=0.1,
    edge_rtol=0.03,
    subsample_ratio=0.01,  # Keep 1% of points
    
    # Output
    output_root=r"D:\Pi3\outputs",
    save_ply=True,
    save_colmap=True,
)

# Run reconstruction
results = reconstruct(config)

# Access results
print(f"\nOutput saved to: {results['output_dir']}")
print(f"Total points: {results['output']['num_points']}")
print(f"Total time: {results['timing']['total_time_s']:.2f}s")
