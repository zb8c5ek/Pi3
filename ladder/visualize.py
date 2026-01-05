"""
Visualization script for Pi3 results using Rerun.

Displays:
- Point cloud with colors
- Camera trajectory
- Input metadata

Usage:
    python visualize.py <output_dir>
    
Example:
    python visualize.py fuse-20-frames_20250105_143022
"""

import argparse
import os
import numpy as np
from pathlib import Path

try:
    import rerun as rr
except ImportError:
    print("Error: rerun library not installed. Install with:")
    print("  pip install rerun-sdk")
    exit(1)

from plyfile import PlyData


def load_ply(filepath):
    """Load point cloud from PLY file."""
    ply_data = PlyData.read(filepath)
    vertex = ply_data['vertex']
    
    points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    
    # Try to load colors
    try:
        colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)
    except (ValueError, IndexError):
        colors = None
    
    return points, colors


def load_trajectory(filepath):
    """Load camera trajectory from NPZ, TXT, or JSON file."""
    import json
    
    if filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)
        camera_poses = data['camera_poses']
        # Extract camera centers
        camera_centers = camera_poses[:, :3, 3]
        return camera_centers, camera_poses
    
    elif filepath.endswith('.txt'):
        # Load from text file (xyz coordinates)
        camera_centers = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    x, y, z = map(float, line.split())
                    camera_centers.append([x, y, z])
        return np.array(camera_centers), None
    
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        camera_centers = np.array(data['camera_centers'])
        camera_poses = np.array(data['camera_poses']) if 'camera_poses' in data else None
        return camera_centers, camera_poses
    
    else:
        raise ValueError(f"Unsupported trajectory file format: {filepath}")


def load_metadata(filepath):
    """Load metadata from text file."""
    metadata = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    return metadata


def visualize_results(output_dir):
    """
    Visualize Pi3 inference results using Rerun.
    
    Args:
        output_dir: Path to the output directory containing results
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        exit(1)
    
    # Load metadata
    metadata_file = output_dir / "metadata.txt"
    metadata = load_metadata(str(metadata_file))
    
    print("=" * 60)
    print("Pi3 Results Visualization")
    print("=" * 60)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    # Load and visualize point cloud
    pointcloud_file = output_dir / "pointcloud.ply"
    if pointcloud_file.exists():
        print(f"\nLoading point cloud from: {pointcloud_file}")
        points, colors = load_ply(str(pointcloud_file))
        print(f"  Points shape: {points.shape}")
        
        # Log point cloud to Rerun
        if colors is not None:
            rr.log(
                "world/pointcloud",
                rr.Points3D(positions=points, colors=colors),
            )
        else:
            rr.log(
                "world/pointcloud",
                rr.Points3D(positions=points),
            )
    else:
        print(f"Warning: Point cloud file not found: {pointcloud_file}")
    
    # Load and visualize camera trajectory
    trajectory_file = None
    for ext in ['.npz', '.json', '.txt']:
        candidate = output_dir / f"trajectory{ext}"
        if candidate.exists():
            trajectory_file = candidate
            break
    
    if trajectory_file:
        print(f"\nLoading camera trajectory from: {trajectory_file}")
        camera_centers, camera_poses = load_trajectory(str(trajectory_file))
        print(f"  Camera trajectory shape: {camera_centers.shape}")
        
        # Log camera trajectory as points with gradient coloring
        num_cameras = len(camera_centers)
        colors = np.zeros((num_cameras, 3), dtype=np.uint8)
        
        # Create a color gradient from blue to red
        for i in range(num_cameras):
            t = i / max(1, num_cameras - 1)
            colors[i, 0] = int(t * 255)      # Red channel
            colors[i, 2] = int((1 - t) * 255)  # Blue channel
        
        rr.log(
            "world/camera_trajectory",
            rr.Points3D(positions=camera_centers, colors=colors, radii=0.05),
        )
        
        # Also log as a line strip to show the trajectory
        rr.log(
            "world/camera_path",
            rr.LineStrips3D([camera_centers], colors=colors),
        )
        
        # Log camera frames for every Nth camera (only if we have pose matrices)
        if camera_poses is not None:
            step = max(1, num_cameras // 10)  # Show at most 10 cameras
            for i in range(0, num_cameras, step):
                pose = camera_poses[i]
                
                # Convert camera-to-world matrix to Rerun format
                from_frame = rr.Transform3D(
                    translation=pose[:3, 3],
                    matrix=pose,
                )
                rr.log(f"world/cameras/camera_{i:03d}", from_frame)
    else:
        print(f"Warning: Trajectory file not found in {output_dir}")
    
    print("\nVisualization opened in Rerun viewer!")
    print("Close the Rerun window to exit.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize Pi3 inference results using Rerun"
    )
    parser.add_argument("output_dir", type=str,
                        help="Path to the output directory containing results")
    parser.add_argument("--server", type=str, default=None,
                        help="Rerun server address (e.g., localhost:9090). If not provided, uses local viewer.")
    
    args = parser.parse_args()
    
    # Initialize Rerun
    if args.server:
        # Connect to remote server
        rr.init("pi3_visualization")
        rr.connect_tcp(args.server)
        print(f"Connected to Rerun server at {args.server}")
    else:
        # Use local spawned viewer
        rr.init("pi3_visualization", spawn=True)
    
    visualize_results(args.output_dir)
