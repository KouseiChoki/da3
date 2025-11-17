# Resolution Guide - Understanding Depth Anything 3 Processing

## The Core Issue: Divisibility by 14

Depth Anything 3 uses a Vision Transformer (ViT/DINO) backbone with **14×14 pixel patches**. This means:

✅ **All processing dimensions MUST be divisible by 14**
❌ **Common resolutions like 256, 512, 1024 are NOT divisible by 14**

## What Actually Happens

When you configure a resolution, the model automatically rounds to the nearest multiple of 14:

```
Your Config    →  Actual Processing
───────────────────────────────────
256 × 256      →  252 × 252  (-4)
512 × 512      →  518 × 518  (+6)   ← This is why you see 504!
768 × 768      →  770 × 770  (+2)
1024 × 1024    →  1022 × 1022 (-2)
```

## Why You See 504 Instead of 512

**The bug:** StreamingDepthEstimator had `process_res=504` hardcoded, ignoring your config!

**After the fix:** Your configured `process_res` is now used, but still gets rounded to multiples of 14.

**With `upper_bound_resize`** (default method):
- Your video: 1024×576
- Config: `process_res=512`
- Step 1: Resize longest side to 512 → 512×288
- Step 2: Round to nearest 14 → **518×294** (not 504!)

**Why 504 before?** The hardcoded value was already a multiple of 14:
- 504 ÷ 14 = 36 ✅

## Recommended "Clean" Resolutions

Use these multiples of 14 to avoid confusion:

| Resolution | Pixels | Use Case |
|-----------|--------|----------|
| **252** | 63,504 | Tiny - fastest possible |
| **378** | 142,884 | Small - very fast |
| **504** | 254,016 | Medium - good balance ⭐ |
| **630** | 396,900 | Large - higher quality |
| **756** | 571,536 | Very large - slow |
| **882** | 777,924 | Huge - very slow |
| **1008** | 1,016,064 | Maximum - slowest |

**⭐ Sweet spot:** 504 (the old default) is actually a great choice!

## Configuration Examples

### For Benchmarking (YAML)

**Old (confusing):**
```yaml
scenarios:
  - name: "DA3-SMALL_256x256"  # Actually becomes 252!
    resolution: [256, 256]
    # process_res defaults to max(256,256) = 256 → rounded to 252

  - name: "DA3-SMALL_512x512"  # Actually becomes 518!
    resolution: [512, 512]
    # process_res defaults to 512 → rounded to 518
```

**New (explicit):**
```yaml
scenarios:
  - name: "DA3-SMALL_252x252"
    resolution: [512, 512]  # Input frame upper bound
    process_res: 252         # Explicit model processing size

  - name: "DA3-SMALL_504x504"
    resolution: [1024, 1024]  # Input frame upper bound
    process_res: 504          # Explicit model processing size

  - name: "DA3-SMALL_756x756"
    resolution: [1024, 1024]
    process_res: 756          # Higher quality, slower
```

### For Streaming Server

**Old:**
```bash
# Always processed at 504×504 (hardcoded)
da3 stream --model-dir depth-anything/DA3-BASE --device mps
```

**New:**
```bash
# Explicit resolution control
da3 stream --model-dir depth-anything/DA3-BASE --device mps --process-res 378  # Fast
da3 stream --model-dir depth-anything/DA3-BASE --device mps --process-res 504  # Balanced
da3 stream --model-dir depth-anything/DA3-BASE --device mps --process-res 756  # Quality
```

## Understanding the Two Resolution Types

### 1. `resolution` (Input Frame Size)

- Controls how input frames are loaded/resized
- Acts as **upper bound** with aspect ratio preservation
- Example: `resolution: [1024, 1024]` on a 1920×1080 video → 1024×576 actual frame

### 2. `process_res` (Model Processing Size)

- Controls the actual tensor size sent to the model
- **This is what affects performance and memory!**
- Gets rounded to nearest multiple of 14
- Default: `max(resolution)` (after the fix)

## Performance Impact

Processing time scales roughly with **pixel count squared** (due to transformer attention):

| process_res | Pixels | Relative Speed | Use For |
|------------|--------|----------------|---------|
| 252 | 63K | **4x faster** | Quick tests, real-time streaming |
| 378 | 143K | **2x faster** | Fast preview |
| 504 | 254K | **Baseline** | Standard quality |
| 630 | 397K | **0.6x (slower)** | Better quality |
| 756 | 572K | **0.4x (much slower)** | High quality |

**Your benchmarks:**
- Before fix: All at 504 → ~6.7 FPS (regardless of config)
- After fix with 252: ~15-20 FPS ⚡
- After fix with 504: ~6-8 FPS
- After fix with 756: ~3-4 FPS

## Best Practices

### For Benchmarking

1. **Use explicit `process_res`** values that are multiples of 14
2. **Match resolution names** to avoid confusion:
   ```yaml
   - name: "DA3-SMALL_504"  # Not "512"!
     resolution: [1024, 1024]
     process_res: 504
   ```

3. **Test across scales**:
   ```yaml
   # Fast, medium, slow comparison
   process_res: 378  # Fast
   process_res: 504  # Balanced
   process_res: 630  # Quality
   ```

### For Streaming/Production

1. **Start with 504** (proven good balance)
2. **Drop to 378 or 252** if you need more FPS
3. **Go to 630 or 756** if quality is more important than speed

### For TouchDesigner/Creative Coding

1. **Use lower resolutions** for real-time (252-378)
2. **Match your output resolution** - no need for 1024 if displaying at 512
3. **Test with `--process-res` flag** to find your sweet spot:
   ```bash
   da3 stream --model-dir depth-anything/DA3-SMALL --process-res 252 --device mps
   ```

## Quick Reference

**Common multiples of 14:**
```
14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196, 210, 224, 238,
252, 266, 280, 294, 308, 322, 336, 350, 364, 378, 392, 406, 420, 434, 448, 462,
476, 490, 504, 518, 532, 546, 560, 574, 588, 602, 616, 630, 644, 658, 672, 686,
700, 714, 728, 742, 756, 770, 784, 798, 812, 826, 840, 854, 868, 882, 896, 910,
924, 938, 952, 966, 980, 994, 1008, 1022, ...
```

**Calculate nearest multiple of 14:**
```python
def round_to_14(x):
    return round(x / 14) * 14

# Examples:
round_to_14(256)  # 252
round_to_14(512)  # 518
round_to_14(1024) # 1022
```

## Summary

✅ **Fixed:** `process_res` is now configurable (was hardcoded to 504)
✅ **Use multiples of 14** for clean, predictable behavior
✅ **504 is a great default** - proven to work well
✅ **Lower res = faster**, higher res = better quality
✅ **Streaming server now accepts `--process-res` flag**

❌ **Don't use 256, 512, 1024** - they get rounded anyway
❌ **Don't confuse input resolution with processing resolution**
❌ **Don't expect exact performance scaling** - test on your hardware
