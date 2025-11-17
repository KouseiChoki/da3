# Streaming and Benchmarking Fixes

## Issues Fixed

### 1. ✅ Aspect Ratio Preservation

**Problem:** Input frames were being squashed to exact square dimensions (256×256, 512×512, etc.), distorting the video.

**Root Cause:** `cv2.resize(frame, resolution)` in `_load_test_frames()` forced exact dimensions.

**Fix:** Changed to aspect-ratio-preserving resize:
```python
# Old (squashes to square):
frame = cv2.resize(frame, resolution)

# New (preserves aspect ratio):
h, w = frame.shape[:2]
target_w, target_h = resolution
scale = min(target_w / w, target_h / h)
new_w = int(w * scale)
new_h = int(h * scale)
frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
```

**Impact:** Video frames now maintain correct aspect ratio with `resolution` as upper bound.

---

### 2. ✅ Resolution Config Now Actually Works

**Problem:** All scenarios showed `torch.Size([12, 3, 504, 504])` regardless of resolution config (256, 512, or 1024). Processing time was identical.

**Root Cause:** The `resolution` field in YAML only affected INPUT frame loading, not MODEL processing. The actual processing resolution was controlled by `process_res` field, which defaulted to 504.

**Fix:** Made `process_res` default to `max(resolution)` instead of hardcoded 504:

```python
# In config.py:
resolution = tuple(scenario_data.get("resolution", [640, 480]))
# Default process_res to max dimension of resolution
default_process_res = max(resolution)

scenario = BenchmarkScenario(
    resolution=resolution,
    process_res=scenario_data.get("process_res", default_process_res),  # Now scales!
    ...
)
```

**Impact:**
- 256×256 config → `process_res=256` → smaller tensors → faster processing
- 512×512 config → `process_res=512` → medium tensors → medium speed
- 1024×1024 config → `process_res=1024` → large tensors → slower processing

**Expected Performance Changes:**
- **256×256**: ~2-3x faster than before (smaller model input)
- **512×512**: Similar to 504 (close to old default)
- **1024×1024**: ~2-4x slower than before (much larger model input)

---

### 3. ✅ Device Logging Added

**Problem:** Device (mps/cuda/cpu) was not shown in benchmark output, making it hard to verify which device was being used.

**Fix:** Added device and process_res to logging output:

```python
print(f"📹 Loaded {len(frames)} test frames")
print(f"🎯 Input Resolution: {frames[0].shape[1]}x{frames[0].shape[0]} (upper bound: {scenario.resolution})")
print(f"⚙️  Model Process Resolution: {scenario.process_res}")
print(f"🔧 Device: {scenario.device}")
print(f"📷 Cameras: {scenario.num_cameras}")
print(f"🔢 Precision: {scenario.precision}")
```

**New Output Example:**
```
📹 Loaded 100 test frames
🎯 Input Resolution: 480x270 (upper bound: (512, 512))
⚙️  Model Process Resolution: 512
🔧 Device: mps
📷 Cameras: 1
🔢 Precision: fp32
```

---

### 4. ✅ Streaming Server DOES Reuse Model Correctly

**Question:** Does the streaming setup reuse the model for each incoming frame?

**Answer:** **YES!** The streaming server (`da3 stream`) correctly reuses the model across ALL frames.

**Evidence:**

1. **Global model instance** (stream_server.py:40-41):
```python
# Global state
stream_estimator: Optional[StreamingDepthEstimator] = None
model: Optional[DepthAnything3] = None
```

2. **Single initialization at startup** (stream_server.py:72-78):
```python
def init_model(model_dir: str, device: str):
    """Initialize the model ONCE."""
    global model, stream_estimator
    print(f"📦 Loading model from {model_dir} on {device}...")
    model = DepthAnything3.from_pretrained(model_dir).to(device)
    stream_estimator = StreamingDepthEstimator(model, device=device)
    print(f"✅ Model loaded and ready for streaming")
```

3. **Reused for every frame** (stream_server.py:148):
```python
# Each frame uses the SAME stream_estimator
depth = stream_estimator.process_frame(frame)
```

**Conclusion:** The streaming server is correctly implemented. Model loads ONCE at startup and processes ALL subsequent frames without reloading.

---

## Performance Investigation

### Why Was Performance Bad?

Based on your log output showing `torch.Size([12, 3, 504, 504])` for all resolutions:

1. **All resolutions were processing at 504×504**
   - 256×256 config was wasting compute (input upscaled to 504)
   - 1024×1024 config was not getting expected quality (input downscaled to 504)

2. **Expected FPS after fix:**
   - **256×256**: 15-20 FPS (was: 6-7 FPS) ← should be much faster now
   - **512×512**: 6-8 FPS (was: 6-7 FPS) ← similar (close to old 504)
   - **1024×1024**: 2-4 FPS (was: 6-7 FPS) ← will be slower but higher quality

### Why Was It Nailed at 6.73 FPS?

The 6.73 FPS you saw was because:
- All scenarios processed at 504×504 (same workload)
- DA3-SMALL forward pass: ~1 second per 12-frame batch
- Window size: 12 frames
- Latency: ~100-150ms

This resulted in consistent ~6-7 FPS across all "resolutions" (which were actually all 504).

After the fix, you should see:
- **Lower resolutions → Higher FPS**
- **Higher resolutions → Lower FPS**
- **Actually different torch.Size() per scenario**

---

## What Changed

### Files Modified

1. **src/depth_anything_3/benchmarking/runner.py**
   - Fixed aspect ratio preservation in `_load_test_frames()`
   - Added device and process_res logging

2. **src/depth_anything_3/benchmarking/config.py**
   - Changed `process_res` default from hardcoded 504 to `max(resolution)`
   - Now scales properly with resolution config

### No Changes Needed

- **stream_server.py** - Already correctly reuses model ✅
- **streaming.py** - StreamingDepthEstimator correctly maintains state ✅
- **Backend service** - Separate from streaming, used only for benchmarks

---

## Testing

### Before Running New Benchmarks

The changes will make benchmarks MORE ACCURATE but DIFFERENT from previous runs:

```bash
# Now this will ACTUALLY process at different resolutions!
da3 benchmark --input-video assets/examples/robot_unitree.mp4 \
  --config benchmarks/example_comprehensive.yaml \
  --device mps
```

**Expected log output:**
```
DA3-SMALL_1cam_256x256_fp32:
  [INFO] Processed Images Shape: torch.Size([12, 3, 256, 256])  ← Now 256!
  [INFO] Model Forward Pass Done. Time: ~0.4s                    ← Faster!
  FPS: ~15-20                                                     ← Higher!

DA3-SMALL_1cam_512x512_fp32:
  [INFO] Processed Images Shape: torch.Size([12, 3, 512, 512])  ← Now 512!
  [INFO] Model Forward Pass Done. Time: ~1.0s                    ← Similar
  FPS: ~6-8                                                       ← Similar

DA3-SMALL_1cam_1024x1024_fp32:
  [INFO] Processed Images Shape: torch.Size([12, 3, 1024, 1024]) ← Now 1024!
  [INFO] Model Forward Pass Done. Time: ~3-4s                     ← SLOWER!
  FPS: ~2-4                                                        ← Lower!
```

### Verify Streaming Server

The streaming server was already correct, but you can verify:

```bash
# Terminal 1: Start streaming server
da3 stream --model-dir depth-anything/DA3-BASE --device mps

# Terminal 2: Send test frames
curl -X POST http://localhost:8080/process/frame \
  -F "file=@test_frame.jpg" \
  -F "format=jpeg" \
  --output depth.jpg

# Model should NOT reload between requests!
```

---

### 5. ✅ Process Resolution Now Configurable in Streaming

**Problem:** `StreamingDepthEstimator` had `process_res=504` hardcoded in line 158, so the scenario config was completely ignored!

**Fix:** Added `process_res` parameter throughout the streaming chain:
- `StreamingDepthEstimator.__init__(process_res=504)`
- Benchmark runner passes `scenario.process_res`
- Stream server CLI accepts `--process-res` flag

**Impact:** Resolution changes NOW actually affect streaming performance!

---

### 6. ✅ Understanding Divisibility by 14

**Important:** The model requires dimensions divisible by 14 (ViT uses 14×14 patches).

**Rounding behavior:**
- 256 → 252 (-4)
- 512 → 518 (+6)
- 1024 → 1022 (-2)
- **504 → 504** (already divisible) ✅

**Recommended resolutions:** 252, 378, 504, 630, 756, 882, 1008

See [RESOLUTION_GUIDE.md](RESOLUTION_GUIDE.md) for complete details.

---

## Summary

| Issue | Status | Impact |
|-------|--------|--------|
| Aspect ratio squashing | ✅ Fixed | Frames no longer distorted |
| Resolution config ignored | ✅ Fixed | Now actually affects processing |
| Device not logged | ✅ Fixed | Now visible in output |
| Streaming model reuse | ✅ Already correct | No changes needed |
| Performance "bad" | ✅ Explained | Was processing all at 504×504 |
| Streaming ignores process_res | ✅ Fixed | Now configurable via parameter |
| Divisibility by 14 | ✅ Documented | Use multiples of 14 for best results |

**Next Run Will Show:**
- Proper aspect ratio in comparison videos
- **Actually different tensor sizes** (252, 518, 1022 instead of all 504)
- **Varying FPS** based on resolution (lower = faster)
- Device clearly shown in logs
- Process resolution explicitly logged
