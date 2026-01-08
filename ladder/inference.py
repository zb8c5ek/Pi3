import torch
import argparse
import numpy as np
import os
import sys
from datetime import datetime

# Setup pi3 path from gin config
from config_utils import setup_pi3_path
config = setup_pi3_path()

from pi3.utils.basic import load_multimodal_data, write_ply, write_camera_trajectory, write_camera_trajectory_txt, write_camera_trajectory_json, write_camera_frustums_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X


def create_output_directory(data_path, output_dir=None):
    """
    Create output directory structure.
    
    If output_dir is not specified, creates a folder named:
    {input_stem}_{datetime}
    
    Returns:
        dict: Paths for point cloud, trajectory.npz, trajectory.ply, and metadata
    """
    if output_dir is None or output_dir == "":
        # Auto-generate output directory
        data_stem = os.path.splitext(os.path.basename(data_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{data_stem}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    output_paths = {
        'pointcloud': os.path.join(output_dir, 'pointcloud.ply'),
        'trajectory': os.path.join(output_dir, 'trajectory.ply'),
        'trajectory_npz': os.path.join(output_dir, 'trajectory.npz'),
        'trajectory_txt': os.path.join(output_dir, 'trajectory.txt'),
        'trajectory_json': os.path.join(output_dir, 'trajectory.json'),
        'metadata': os.path.join(output_dir, 'metadata.txt'),
        'directory': output_dir,
    }
    
    return output_paths


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model and save point cloud + camera trajectory.")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input image directory or a video file.")
    
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. If not specified, auto-generates as {input_stem}_{datetime}")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # Create output directory structure
    output_paths = create_output_directory(args.data_path, args.output_dir)
    print(f"Output directory: {output_paths['directory']}")

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 

    """
    Args:
        imgs (torch.Tensor): Input RGB images valued in [0, 1].
            Shape: (B, N, 3, H, W).
        intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
            Shape: (B, N, 3, 3).
            Values are in pixel coordinates (not normalized).
        rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
            Shape: (B, N, H, W, 3).
            Can replace `intrinsics` as a geometric condition.
        poses (torch.Tensor, optional): Camera-to-World matrices.
            Shape: (B, N, 4, 4).
            Coordinate system: OpenCV convention (Right-Down-Forward).
        depths (torch.Tensor, optional): Ground truth or prior depth maps.
            Shape: (B, N, H, W).
            Invalid values (e.g., sky or missing data) should be set to 0.
    """

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 4. Process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save pure point cloud (without camera markers)
    print(f"Saving pure point cloud to: {output_paths['pointcloud']}")
    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], 
              output_paths['pointcloud'])
    
    # 6. Save camera trajectory PLY with frustums
    print(f"Saving trajectory with frustums to: {output_paths['trajectory']}")
    write_camera_frustums_ply(res['camera_poses'], 
                              res['points'][0][masks].cpu(),
                              output_paths['trajectory'])
    
    # 7. Save camera trajectory in multiple formats
    print(f"Saving trajectory (NPZ) to: {output_paths['trajectory_npz']}")
    write_camera_trajectory(res['camera_poses'], output_paths['trajectory_npz'])
    
    print(f"Saving trajectory (TXT) to: {output_paths['trajectory_txt']}")
    write_camera_trajectory_txt(res['camera_poses'], output_paths['trajectory_txt'])
    
    print(f"Saving trajectory (JSON) to: {output_paths['trajectory_json']}")
    write_camera_trajectory_json(res['camera_poses'], output_paths['trajectory_json'])
    
    # 8. Save metadata
    with open(output_paths['metadata'], 'w') as f:
        f.write(f"Input data path: {os.path.abspath(args.data_path)}\n")
        f.write(f"Output directory: {os.path.abspath(output_paths['directory'])}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Sampling interval: {args.interval}\n")
        f.write(f"Device: {args.device}\n")
        if args.ckpt:
            f.write(f"Checkpoint: {args.ckpt}\n")
    
    print(f"Done! Results saved to: {output_paths['directory']}")

