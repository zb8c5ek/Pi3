# Using COLMAP Camera Poses with Pi3X

This guide shows you how to use pre-computed COLMAP camera poses as priors for Pi3X to get dense depth reconstruction.

## Overview

**Pi3X** supports **multimodal conditioning**, which means you can provide:
- **Camera poses** (from COLMAP) - Camera-to-World 4x4 matrices
- **Camera intrinsics** (from COLMAP) - 3x3 K matrices  
- **Depth priors** (optional) - If you have any partial depth data

When you provide these as conditions, Pi3X will:
1. Use the camera poses to constrain the reconstruction
2. Predict **dense depth maps** for all frames
3. Generate a **metric-scale 3D point cloud** (approximate metric scale)

## Step-by-Step Workflow

### 1. Run COLMAP on Your Images

If you haven't already, run COLMAP to get camera poses:

```bash
# Automatic reconstruction
colmap automatic_reconstructor \
    --workspace_path /path/to/workspace \
    --image_path /path/to/images

# Or manual pipeline
colmap feature_extractor --database_path database.db --image_path images/
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images/ --output_path sparse/
```

This will create `sparse/0/` containing:
- `cameras.txt` - Camera intrinsics
- `images.txt` - Camera poses (world-to-camera)
- `points3D.txt` - Sparse 3D points

### 2. Convert COLMAP Format to Pi3X Format

COLMAP uses **world-to-camera (W2C)** matrices, but Pi3X expects **camera-to-world (C2W)** matrices in OpenCV convention.

**Option A: Using the provided conversion script**

```bash
# Convert COLMAP data to Pi3X format
python convert_colmap_to_pi3x.py \
    --colmap_path sparse/0 \
    --output_path conditions.npz
```

This creates `conditions.npz` containing:
- `poses`: (N, 4, 4) - Camera-to-world matrices in OpenCV format
- `intrinsics`: (N, 3, 3) - Camera intrinsic matrices
- `image_names`: (N,) - Image filenames for reference

**Option B: If you have binary COLMAP files**

First convert to text format:
```bash
colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0 \
    --output_type TXT
```

Then run the conversion script as above.

### 3. Run Pi3X with Camera Pose Priors

Now use Pi3X with your COLMAP poses:

```bash
python example_mm.py \
    --data_path /path/to/images \
    --conditions_path conditions.npz \
    --save_path output_with_colmap.ply \
    --interval 1
```

**Parameters:**
- `--data_path`: Your image directory (same images used in COLMAP)
- `--conditions_path`: The converted conditions.npz file
- `--save_path`: Output point cloud file
- `--interval`: Sampling interval (1 = use all images)

### 4. Compare Results

You can compare reconstruction with and without COLMAP priors:

```bash
# WITH COLMAP priors (better metric scale and accuracy)
python example_mm.py \
    --data_path images/ \
    --conditions_path conditions.npz \
    --save_path output_with_colmap.ply

# WITHOUT priors (pure Pi3X prediction)
python example_mm.py \
    --data_path images/ \
    --save_path output_no_colmap.ply
```

The version with COLMAP priors should give you:
- ✓ More accurate metric scale
- ✓ Better camera alignment
- ✓ Denser depth maps (Pi3X predicts dense depth, COLMAP gives sparse)
- ✓ More stable reconstruction

## Understanding the Data Format

### Camera Poses
- **Shape**: (N, 4, 4)
- **Format**: Camera-to-world transformation matrices
- **Convention**: OpenCV (Right-Down-Forward)
- **Example**:
```python
pose = [
    [r11, r12, r13, tx],
    [r21, r22, r23, ty],
    [r31, r32, r33, tz],
    [0,   0,   0,   1 ]
]
```

### Intrinsics
- **Shape**: (N, 3, 3)
- **Format**: Standard camera intrinsic matrix
- **Example**:
```python
K = [
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
]
```

### Depths (Optional)
- **Shape**: (N, H, W)
- **Format**: Depth values in meters (or same unit as camera poses)
- **Note**: Set invalid pixels (e.g., sky) to 0

## Advanced: Partial Conditioning

Pi3X supports **partial conditioning** - you don't need to provide poses for ALL frames:

```python
import numpy as np

# Example: Only condition on every 10th frame
data = np.load('conditions.npz')
poses = data['poses']
intrinsics = data['intrinsics']

# Create masks to control which frames get conditioned
N = len(poses)
mask_add_pose = np.zeros((N, N), dtype=bool)

# Only use poses for frames 0, 10, 20, 30, ...
keyframes = range(0, N, 10)
for i in keyframes:
    mask_add_pose[i, :] = True
    mask_add_pose[:, i] = True

# Save with masks
np.savez('conditions_partial.npz',
         poses=poses,
         intrinsics=intrinsics,
         mask_add_pose=mask_add_pose)
```

## Tips for Best Results

1. **Image Quality**: Use the same images for COLMAP and Pi3X
2. **Frame Selection**: If you have many frames, subsample consistently
3. **Scale**: COLMAP scale is arbitrary - Pi3X will give approximately metric scale when conditioned
4. **Memory**: For long sequences, process in chunks or use lower resolution
5. **Confidence Filtering**: Adjust the confidence threshold in example_mm.py (default 0.1)

## Output Visualization

The output `.ply` file can be viewed with:
- **CloudCompare** (recommended for large point clouds)
- **MeshLab**
- **Open3D** (Python library)

```python
# Quick visualization with Open3D
import open3d as o3d
pcd = o3d.io.read_point_cloud('output_with_colmap.ply')
o3d.visualization.draw_geometries([pcd])
```

## Troubleshooting

### Issue: "Binary COLMAP files not supported"
**Solution**: Convert to text format using `colmap model_converter`

### Issue: Point cloud has wrong scale
**Solution**: 
- Check COLMAP scale (it's arbitrary)
- Pi3X provides approximate metric scale when conditioned
- You may need to rescale COLMAP poses to match your desired metric

### Issue: Image order mismatch
**Solution**: The conversion script sorts images by COLMAP image_id. Make sure your `--data_path` directory lists images in the same order, or check `image_names` in the .npz file.

### Issue: Out of memory
**Solution**:
- Reduce `--interval` to process fewer frames
- Lower the `PIXEL_LIMIT` in the code
- Process sequence in chunks

## Example: Complete Pipeline

```bash
# 1. Run COLMAP
colmap automatic_reconstructor \
    --workspace_path ./colmap_workspace \
    --image_path ./images

# 2. Convert to text format (if needed)
colmap model_converter \
    --input_path ./colmap_workspace/sparse/0 \
    --output_path ./colmap_workspace/sparse/0 \
    --output_type TXT

# 3. Convert to Pi3X format
python convert_colmap_to_pi3x.py \
    --colmap_path ./colmap_workspace/sparse/0 \
    --output_path conditions.npz

# 4. Run Pi3X with COLMAP priors
python example_mm.py \
    --data_path ./images \
    --conditions_path conditions.npz \
    --save_path dense_reconstruction.ply

# 5. View result
# Open dense_reconstruction.ply in CloudCompare or MeshLab
```

## What You Get

**COLMAP alone**: Sparse 3D points (thousands to millions of points)
**Pi3X with COLMAP priors**: Dense 3D reconstruction (millions to billions of points) with:
- Dense depth for every pixel in every frame
- Metric-scale point cloud
- Better accuracy due to pose constraints
- Smooth surfaces and complete geometry

This combines the best of both: COLMAP's accurate camera poses + Pi3X's dense depth prediction!
