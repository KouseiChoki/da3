# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import imageio
import numpy as np
import cv2
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth
from depth_anything_3.utils.fileutils import mvwrite
import glob
import re

def get_next_index(folder):
    """扫描已有文件编号，返回下一个编号"""
    files = glob.glob(os.path.join(folder, "*.jpg"))  # 或 *.exr
    if not files:
        return 0
    nums = []
    for f in files:
        name = os.path.basename(f)
        m = re.match(r"(\d+)", name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 0



def export_to_exr(
    prediction: Prediction,
    export_dir: str,
):
    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "depths"), exist_ok=True)
    start_idx_img = get_next_index(os.path.join(export_dir, "images"))
    for idx in range(prediction.depth.shape[0]):
        global_idx = idx+start_idx_img
        image_save_path = os.path.join(export_dir, f"images/{global_idx:04d}.jpg")
        depth_save_path = os.path.join(export_dir, f"depths/{global_idx:04d}.exr")
        depth = prediction.depth[idx]
        depth = np.repeat(depth[...,None],4,axis=2)
        depth = cv2.resize(depth,(1920,1080))
        d = depth[...,0]
        d = (d - d.min()) / (d.max() - d.min())
        # if args.norm:
        depth[...,-1] = d
        mvwrite(depth_save_path,depth,precision='half')
        image_vis = images_u8[idx]
        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.resize(image_vis,(1920,1080))
        imageio.imwrite(image_save_path, image_vis)

