# Depth Anything 3 - Streaming Benchmarking System

## Overview

A comprehensive benchmarking system for testing and comparing streaming depth estimation performance across different models, configurations, and hardware setups.

## Features

✅ **Multi-Model Testing**: Compare DA3-SMALL, DA3-BASE, and DA3-LARGE side-by-side
✅ **Configuration Variations**: Test different resolutions, precisions (fp16/fp32), and multi-camera setups
✅ **Comprehensive Metrics**: FPS, latency (avg/p50/p95/p99), memory usage, frame-level statistics
✅ **Visual Outputs**: Comparison videos, depth map exports, frame-by-frame visualizations
✅ **Interactive Reports**: HTML reports with Chart.js visualizations and detailed breakdowns
✅ **YAML Configuration**: Easy-to-edit configuration files with sensible defaults
✅ **CLI Integration**: Simple command-line interface with `da3 benchmark`

## Quick Start

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Quick Benchmark

Test all three models with default settings (~15 minutes):

```bash
da3 benchmark --config benchmarks/example_quick.yaml
```

### 3. View Results

Open the generated HTML report:

```bash
open benchmark_results/quick/report.html
```

## Architecture

### Core Components

```
src/depth_anything_3/benchmarking/
├── __init__.py          # Package exports
├── config.py            # Configuration system (YAML parsing, scenario definitions)
├── metrics.py           # Metrics collection (FPS, latency, memory tracking)
├── runner.py            # Benchmark execution engine
└── report.py            # HTML report generation with Chart.js
```

### Data Flow

```
Config (YAML)
    ↓
BenchmarkRunner
    ↓
For each scenario:
    - Load model
    - Load test frames
    - Run warmup
    - Process frames (collect metrics)
    - Save depth maps & videos
    - Generate metrics.json
    ↓
ReportGenerator
    ↓
HTML Report (report.html)
```

## Configuration System

### Scenario Structure

Each benchmark scenario tests a specific configuration:

```yaml
- name: "DA3-BASE_1cam_640x480_fp32"

  # Model
  model:
    name: "DA3-BASE"
    model_dir: "depth-anything/DA3-BASE"

  # Input
  num_cameras: 1                    # 1-3 cameras
  resolution: [640, 480]            # Input resolution

  # Processing
  precision: "fp32"                 # fp16 or fp32
  device: "mps"                     # mps, cuda, cpu
  process_res: 504                  # Internal processing resolution

  # Streaming
  stream_config:
    window_size: null               # Auto-detect based on device
    overlap: null
    buffer_size: null
    output_latency_frames: 2

  # Testing
  num_frames: 100                   # Frames to process
  warmup_frames: 10                 # Warmup (excluded from metrics)

  # Output
  save_depth_maps: true
  save_comparison_video: true
```

### Global Configuration

```yaml
name: "My Benchmark"
description: "Optional description"

# Test data
input_video: "path/to/video.mp4"   # Optional: use real video
input_images: "path/to/images/"    # Optional: use image sequence

# Output
output_dir: "benchmark_results"

# Comparison
compare_against_baseline: true
baseline_scenario_name: "DA3-BASE_1cam_640x480_fp32"
```

## Metrics Collected

### Performance Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Average FPS** | Mean frames processed per second | fps |
| **Min/Max FPS** | FPS range across all frames | fps |
| **Std FPS** | FPS standard deviation | fps |
| **Average Latency** | Mean time per frame | ms |
| **P50 Latency** | Median latency | ms |
| **P95 Latency** | 95th percentile latency | ms |
| **P99 Latency** | 99th percentile latency | ms |

### Memory Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **Average Memory** | Mean GPU/MPS memory used | MB |
| **Peak Memory** | Maximum memory allocated | MB |
| **Peak Allocated** | Peak allocation during run | MB |

### System Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **CPU Usage** | CPU utilization percentage | % |
| **RAM Used** | System RAM usage | GB |

### Frame-Level Metrics

For each frame:
- Frame index
- Inference time (model forward pass only)
- Total time (including pre/post-processing)
- Memory used at that moment

## Output Structure

```
benchmark_results/
├── report.html                          # 📄 Main HTML report
│
├── DA3-SMALL_1cam_640x480_fp32/
│   ├── metrics.json                     # 📊 Raw metrics
│   ├── comparison.mp4                   # 🎥 Side-by-side video
│   └── depth_maps/                      # 🖼️  Individual depth maps
│       ├── depth_0000.png
│       ├── depth_0001.png
│       └── ...
│
├── DA3-BASE_1cam_640x480_fp32/
│   └── ...
│
└── DA3-LARGE_1cam_640x480_fp32/
    └── ...
```

## Usage Examples

### Create Default Config

Generate a template to customize:

```bash
da3 benchmark --create-default --output-dir my_benchmark
```

Edit `my_benchmark/benchmark_config.yaml`, then run:

```bash
da3 benchmark --config my_benchmark/benchmark_config.yaml
```

### Quick Test with Video

Test without creating a config file:

```bash
da3 benchmark \
  --input-video path/to/video.mp4 \
  --device mps \
  --num-frames 50
```

This creates a default config testing all models with your video.

### Run Example Configs

**Quick benchmark** (~15 min):
```bash
da3 benchmark --config benchmarks/example_quick.yaml
```

**Comprehensive benchmark** (~1-2 hours):
```bash
da3 benchmark --config benchmarks/example_comprehensive.yaml
```

### Override Config Parameters

```bash
# Override device
da3 benchmark --config config.yaml --device cuda

# Override input video
da3 benchmark --config config.yaml --input-video new_video.mp4

# Change output directory
da3 benchmark --config config.yaml --output-dir results_v2
```

## Example Configurations

### [benchmarks/example_quick.yaml](benchmarks/example_quick.yaml)

- **Purpose**: Quick model comparison
- **Time**: ~15 minutes
- **Scenarios**: 3 (SMALL, BASE, LARGE)
- **Frames**: 50 per scenario

### [benchmarks/example_comprehensive.yaml](benchmarks/example_comprehensive.yaml)

- **Purpose**: Full configuration matrix
- **Time**: ~1-2 hours
- **Scenarios**: 13 (all combinations)
- **Frames**: 100 per scenario
- **Tests**: Resolutions, precisions, multi-camera

## HTML Report Features

The generated HTML report includes:

### 1. Performance Summary Table

Quick comparison of all scenarios with key metrics.

### 2. Interactive Charts (Chart.js)

- **Average FPS Comparison**: Bar chart comparing throughput
- **Latency Comparison**: Avg and P95 latency side-by-side
- **Memory Usage**: Avg and peak memory comparison
- **Frame Timeline**: Line chart showing per-frame processing time

### 3. Detailed Results

Per-scenario breakdowns:
- Min/max FPS
- Latency percentiles (P50, P95, P99)
- Memory statistics
- System resource usage

### 4. Responsive Design

- Mobile-friendly layout
- Gradient header
- Clean, modern styling
- Accessible color scheme

## Performance Expectations

### Apple Silicon (M1/M2/M3) @ 640×480

| Model | FPS | Latency | Memory |
|-------|-----|---------|--------|
| **DA3-SMALL** | 20-30 | 30-50ms | ~1.5GB |
| **DA3-BASE** | 10-20 | 50-100ms | ~2GB |
| **DA3-LARGE** | 5-10 | 100-200ms | ~3GB |

### NVIDIA RTX 3080 @ 640×480

| Model | FPS | Latency | Memory |
|-------|-----|---------|--------|
| **DA3-SMALL** | 40-60 | 15-25ms | ~1.5GB |
| **DA3-BASE** | 20-40 | 25-50ms | ~2GB |
| **DA3-LARGE** | 10-20 | 50-100ms | ~3GB |

*Note: Actual performance varies based on hardware, thermal conditions, and system load.*

## Best Practices

### 1. Start with Quick Test

Always run `example_quick.yaml` first to verify setup and get baseline expectations.

### 2. Use Real Test Data

Provide your own video for realistic results:

```yaml
input_video: "path/to/typical_use_case.mp4"
```

Synthetic frames (default) are less representative of real-world performance.

### 3. Choose Appropriate Frame Count

- **Quick validation**: 50 frames
- **Standard benchmark**: 100 frames
- **Statistical significance**: 200+ frames

### 4. Platform-Specific Settings

**macOS (Apple Silicon)**:
```yaml
device: "mps"
precision: "fp32"  # fp16 NOT supported on MPS - will auto-fallback to fp32
```

**Linux/Windows (NVIDIA GPU)**:
```yaml
device: "cuda"
precision: "fp16"  # Significant speedup on Ampere+ (RTX 30/40 series)
precision: "fp32"  # For older GPUs or maximum accuracy
```

**⚠️ Important**: fp16 precision is **only supported on CUDA devices**. If you specify fp16 on MPS or CPU, the benchmark runner will automatically fall back to fp32 with a warning.

### 5. Monitor Memory Usage

Multi-camera and high-resolution tests use more memory:

- 1 camera @ 640×480: ~2GB
- 2 cameras @ 640×480: ~4GB
- 3 cameras @ 640×480: ~6GB
- 1 camera @ 1280×720: ~3GB

Reduce parameters if encountering OOM errors.

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: Crash with memory allocation error

**Solutions**:
1. Reduce `num_cameras` (3 → 2 → 1)
2. Lower `resolution` (1280×720 → 640×480)
3. Use smaller model (LARGE → BASE → SMALL)
4. Decrease `num_frames`

### Slow Performance

**Symptoms**: FPS much lower than expected

**Checks**:
1. Verify correct device (`mps` for Apple Silicon)
2. Close GPU-intensive applications
3. Check thermal throttling (system monitor)
4. Ensure warmup completed (first few frames are slow)

### Import Errors

**Symptoms**: `ModuleNotFoundError`

**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

Required for benchmarking:
- `psutil` (system metrics)
- `pyyaml` (config parsing)
- `tqdm` (progress bars)

## Integration with Streaming

The benchmarking system uses the same `StreamingDepthEstimator` as the production streaming server, ensuring:

✅ **Accurate Performance**: Benchmarks reflect real streaming behavior
✅ **Configuration Parity**: Same settings work in both contexts
✅ **Realistic Testing**: Sliding window processing matches production

## Future Enhancements

Potential additions:

- [ ] Quality metrics (depth accuracy, consistency)
- [ ] Multi-GPU benchmarking
- [ ] Batch processing mode comparison
- [ ] Network latency simulation
- [ ] Power consumption tracking
- [ ] Comparison against baseline checkpoints
- [ ] Automated regression detection

## Related Documentation

- [CLAUDE.md](CLAUDE.md): Main project documentation
- [benchmarks/README.md](benchmarks/README.md): Detailed benchmark guide
- [src/depth_anything_3/services/streaming.py](src/depth_anything_3/services/streaming.py): Streaming architecture

## Contributing

To add new metrics or features:

1. **Metrics**: Edit [metrics.py](src/depth_anything_3/benchmarking/metrics.py)
2. **Charts**: Update [report.py](src/depth_anything_3/benchmarking/report.py)
3. **Config options**: Extend [config.py](src/depth_anything_3/benchmarking/config.py)
4. **Example configs**: Add to [benchmarks/](benchmarks/)

## License

Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
Licensed under Apache License 2.0
