# Pose Estimation Benchmarking

## Overview

**Current Status**: ⚠️ **Configuration added, implementation pending**

The benchmarking system now supports configuration for pose estimation testing, but the runner implementation is not yet complete. This document explains:

1. What pose estimation features DA3 supports
2. How to configure pose estimation benchmarks
3. What needs to be implemented

## DA3 Pose Estimation Capabilities

According to the [official repo](https://github.com/ByteDance-Seed/Depth-Anything-3), DA3 can:

### 1. **Camera Pose Estimation**
- Estimates camera extrinsics (world-to-camera transform, 4×4 matrix)
- Estimates camera intrinsics (focal length, principal point, 3×3 matrix)
- Works from single or multiple images

### 2. **Pose-Conditioned Depth**
- Accepts camera poses as input
- Uses poses to improve depth consistency across frames
- Particularly useful for video sequences with camera motion

### 3. **Multi-View Consistency**
- When poses are provided, depth maps are more geometrically consistent
- Enables better 3D reconstruction
- Reduces drift and scale ambiguity

## Configuration Added

### New BenchmarkScenario Fields

```yaml
scenarios:
  - name: "My_Pose_Test"
    # ... standard fields ...

    # Pose estimation configuration
    test_pose_estimation: true        # Enable pose estimation
    test_pose_conditioned: true       # Use estimated poses for depth
    align_to_input_ext_scale: true    # Align depth to pose scale
    save_pose_visualization: true     # Save camera trajectory viz
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `test_pose_estimation` | bool | `false` | Test camera pose estimation |
| `test_pose_conditioned` | bool | `false` | Use poses to condition depth estimation |
| `align_to_input_ext_scale` | bool | `false` | Align prediction scale to input poses |
| `save_pose_visualization` | bool | `false` | Save 3D camera trajectory plot |

## Example Configuration

See [benchmarks/example_pose_estimation.yaml](benchmarks/example_pose_estimation.yaml) for a complete example.

### Scenario Comparison

The example config tests 4 scenarios:

1. **Baseline** (`test_pose_estimation: false`)
   - Standard depth estimation
   - No pose information
   - Fastest, but less consistent

2. **Pose Estimation** (`test_pose_estimation: true`)
   - Estimates camera poses
   - Doesn't use them for depth yet
   - Tests pose estimation accuracy

3. **Pose-Conditioned** (`test_pose_conditioned: true`)
   - Estimates poses AND uses them
   - Improves depth consistency
   - Best quality, slightly slower

4. **Small Model** (DA3-SMALL with pose conditioning)
   - Faster inference
   - Comparable pose accuracy
   - Good for real-time applications

## Implementation TODO

### ⚠️ Runner Implementation Needed

The `BenchmarkRunner` needs to be updated to:

1. **Call API with Pose Estimation**
   ```python
   # Current (no pose):
   prediction = model.inference(
       image=frames,
       export_dir=None,
       process_res=504,
   )

   # Needed (with pose estimation):
   prediction = model.inference(
       image=frames,
       export_dir=None,
       process_res=504,
       # Model will estimate these:
       extrinsics=None,  # Will be populated by model
       intrinsics=None,   # Will be populated by model
   )
   ```

2. **Handle Pose-Conditioned Mode**
   ```python
   if scenario.test_pose_conditioned:
       # First pass: estimate poses
       pose_prediction = model.inference(image=frames, ...)
       estimated_extrinsics = pose_prediction.extrinsics
       estimated_intrinsics = pose_prediction.intrinsics

       # Second pass: use estimated poses for better depth
       final_prediction = model.inference(
           image=frames,
           extrinsics=estimated_extrinsics,
           intrinsics=estimated_intrinsics,
           align_to_input_ext_scale=scenario.align_to_input_ext_scale,
       )
   ```

3. **Collect Pose Metrics**
   - Track pose estimation time
   - Measure pose consistency (frame-to-frame)
   - Compute camera trajectory statistics

4. **Save Pose Visualizations**
   - 3D camera trajectory plot
   - Camera frustums in 3D space
   - Pose evolution over time

### Metrics to Track

Additional metrics needed in `BenchmarkMetrics`:

- `pose_estimation_time_ms`: Time to estimate poses
- `avg_translation_change`: Average camera translation between frames
- `avg_rotation_change`: Average camera rotation between frames
- `pose_consistency`: Smoothness of estimated trajectory
- `depth_consistency_with_pose`: Improvement vs non-pose-conditioned

### Visualization Needed

Create `_save_pose_visualization()` method:

```python
def _save_pose_visualization(
    self,
    extrinsics: np.ndarray,  # (N, 4, 4)
    output_dir: Path,
):
    """
    Save 3D visualization of camera trajectory.

    Creates:
    - trajectory.png: Top-down view of camera path
    - cameras_3d.html: Interactive 3D view with camera frustums
    - pose_stats.txt: Statistics about camera motion
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Extract camera positions
    positions = extrinsics[:, :3, 3]  # (N, 3)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Camera Path')

    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=range(len(positions)), cmap='viridis', s=50)

    # ... styling, save, etc.
```

## Expected Results

### Without Pose Conditioning

```
Benchmark Results: DA3-BASE_standard
====================================
Total Frames: 50
Avg FPS: 8.2

Depth consistency: Medium
Temporal stability: Medium
```

### With Pose Conditioning

```
Benchmark Results: DA3-BASE_pose_conditioned
=============================================
Total Frames: 50
Avg FPS: 7.5 (slightly slower due to pose estimation)

Pose Estimation:
  Avg translation: 0.15m per frame
  Avg rotation: 2.3° per frame
  Trajectory smoothness: High

Depth consistency: High ✅
Temporal stability: High ✅
```

## Use Cases

### When to Use Pose Estimation

**✅ Good For**:
- Video sequences with camera motion
- Multi-view reconstruction
- SLAM/odometry applications
- Consistent depth across frames

**❌ Not Needed For**:
- Single static images
- When you already have ground truth poses (e.g., COLMAP)
- Real-time streaming where latency is critical

### When to Use Pose-Conditioned Depth

**✅ Good For**:
- Improving depth consistency in videos
- 3D reconstruction pipelines
- Applications needing metric scale
- Multi-frame depth fusion

**❌ Not Needed For**:
- Single-frame depth estimation
- Applications that don't care about absolute scale
- When poses are unreliable

## Integration with Existing Features

### Streaming Mode

Pose estimation can be integrated with streaming:

```python
# StreamingDepthEstimator with pose
estimator = StreamingDepthEstimator(
    model=model,
    device="mps",
    enable_pose_estimation=True,  # TODO: Add this parameter
    use_pose_conditioning=True,
)

for frame in video_frames:
    depth = estimator.process_frame(frame)
    # Depth is now pose-conditioned for better consistency
```

### COLMAP Integration

When ground truth poses are available:

```bash
# Benchmark with COLMAP poses
da3 benchmark \
  --config pose_benchmark.yaml \
  --colmap-dir path/to/colmap \
  --compare-estimated-vs-colmap  # Compare estimated vs GT poses
```

## Testing the Configuration

### Quick Test (Without Implementation)

```bash
# This will load the config but won't actually run pose estimation yet
da3 benchmark --config benchmarks/example_pose_estimation.yaml --create-default

# Verify configuration is valid
cat benchmark_results/pose_estimation/benchmark_config.yaml
```

### When Implementation is Complete

```bash
# Run full pose estimation benchmark
da3 benchmark --config benchmarks/example_pose_estimation.yaml

# Expected outputs:
# - benchmark_results/pose_estimation/
#   ├── DA3-BASE_standard/           (no pose)
#   ├── DA3-BASE_pose_estimation/    (pose estimated)
#   │   ├── trajectory.png
#   │   └── pose_stats.txt
#   ├── DA3-BASE_pose_conditioned/   (pose used for depth)
#   │   ├── trajectory.png
#   │   ├── comparison.mp4           (better consistency)
#   │   └── depth_maps/
#   └── report.html                  (includes pose metrics)
```

## Next Steps

To complete pose estimation benchmarking:

1. ✅ **Configuration** - Done! Fields added to config system
2. ✅ **Example config** - Done! See `example_pose_estimation.yaml`
3. ⏳ **Runner implementation** - TODO: Update `runner.py`
4. ⏳ **Metrics collection** - TODO: Add pose-specific metrics
5. ⏳ **Visualization** - TODO: Implement trajectory plotting
6. ⏳ **Report generation** - TODO: Add pose metrics to HTML report

## References

- [Depth Anything 3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [DA3 API Documentation](src/depth_anything_3/api.py)
- [Pose Alignment Utils](src/depth_anything_3/utils/pose_align.py)
- [Example Pose Config](benchmarks/example_pose_estimation.yaml)

## Summary

✅ **Configuration ready**: You can now define pose estimation benchmarks in YAML
✅ **Example provided**: See `example_pose_estimation.yaml` for complete example
⏳ **Implementation pending**: Runner needs to be updated to actually use these settings

The configuration foundation is in place. When you're ready to implement full pose estimation benchmarking, the structure is ready to support it!
