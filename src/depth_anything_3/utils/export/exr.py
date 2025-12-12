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
import fbx

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

def w2c_to_sixdof(extr):
        """
        extr: 3x4, OpenCV world-to-camera
        
        returns:
            pos: (x,y,z)
            rot: (rx,ry,rz) degree, XYZ rotation, FBX style: x-right, y-up, z-back
        """

        # 1) 构造 4x4 W2C
        W2C = np.eye(4)
        W2C[:3, :4] = extr

        # 2) 转为 C2W
        C2W = np.linalg.inv(W2C)

        # 3) OpenCV camera frame 变 FBX frame（Z 翻转）
        #    OpenCV: X right, Y down, Z forward
        #    FBX:    X right, Y up, Z back
        cv2_to_fbx = np.diag([1, -1, -1])  # 翻转 Y 和 Z
        T = np.eye(4)
        T[:3, :3] = cv2_to_fbx

        C2W_fbx = C2W @ T

        # ===== Position =====
        pos = C2W_fbx[:3, 3]

        # ===== Rotation (XYZ 欧拉角) =====
        R = C2W_fbx[:3, :3]
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            ry = np.degrees(np.arctan2(-R[2, 0], sy))
            rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        else:
            rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            ry = np.degrees(np.arctan2(-R[2, 0], sy))
            rz = 0.0

        rot = np.array([rx, ry, rz])

        return pos, rot

def export_cameras_to_fbx(prediction, export_dir):
    fbx_path = f"{export_dir}/cameras.fbx"

    # 初始化 FBX
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)
    scene = fbx.FbxScene.Create(manager, "Scene")

    root = scene.GetRootNode()
    cam_name = f"Camera"
    camera = fbx.FbxCamera.Create(scene, cam_name)
    cam_node = fbx.FbxNode.Create(scene, cam_name)
    # 遍历每帧
    for idx in range(prediction.extrinsics.shape[0]):
        extr = prediction.extrinsics[idx]   # 3x4
        intr = prediction.intrinsics[idx]   # 3x3 (你没用到，保留未来扩展 FOV)

        pos, rot = w2c_to_sixdof(extr)

        # 创建相机节点
        
        cam_node.SetNodeAttribute(camera)

        # 设置 Transform
        cam_node.LclTranslation.Set(fbx.FbxDouble3(pos[0], pos[1], pos[2]))
        cam_node.LclRotation.Set(fbx.FbxDouble3(rot[0], rot[1], rot[2]))

        # 加入场景
        root.AddChild(cam_node)

    # 保存 FBX
    exporter = fbx.FbxExporter.Create(manager, "")
    if not exporter.Initialize(fbx_path, -1, manager.GetIOSettings()):
        print("❌ FBX 初始化失败")
        return None

    exporter.Export(scene)
    exporter.Destroy()
    print(f"✅ FBX saved to {fbx_path}")

    return fbx_path

def export_camera_anim_fbx(prediction, export_path, fps=24):
    manager = fbx.FbxManager.Create()
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    scene = fbx.FbxScene.Create(manager, "CameraScene")
    scene.GetGlobalSettings().SetTimeMode(fbx.FbxTime.EMode.eFrames24)
    root = scene.GetRootNode()

    camera = fbx.FbxCamera.Create(scene, "MainCamera")
    camera_node = fbx.FbxNode.Create(scene, "MainCamera")
    camera_node.SetNodeAttribute(camera)
    root.AddChild(camera_node)

    anim_stack = fbx.FbxAnimStack.Create(scene, "CameraAnimation")
    anim_layer = fbx.FbxAnimLayer.Create(scene, "BaseLayer")
    anim_stack.AddMember(anim_layer)

    tx = camera_node.LclTranslation.GetCurve(anim_layer, "X", True)
    ty = camera_node.LclTranslation.GetCurve(anim_layer, "Y", True)
    tz = camera_node.LclTranslation.GetCurve(anim_layer, "Z", True)

    rx = camera_node.LclRotation.GetCurve(anim_layer, "X", True)
    ry = camera_node.LclRotation.GetCurve(anim_layer, "Y", True)
    rz = camera_node.LclRotation.GetCurve(anim_layer, "Z", True)

    num_frames = prediction.extrinsics.shape[0]

    t = fbx.FbxTime()
    t.SetGlobalTimeMode(fbx.FbxTime.EMode.eFrames24) # Set to fps=60

    for frame_idx in range(num_frames):
        extr = prediction.extrinsics[frame_idx]
        pos, rot = w2c_to_sixdof(extr)

        # 🔥 100% 兼容所有 SDK 的帧时间写法
        t.SetFrame(frame_idx, fbx.FbxTime.EMode.eFrames24)
        # position
        tx.KeySetValue(tx.KeyAdd(t)[0], pos[0])
        ty.KeySetValue(ty.KeyAdd(t)[0], pos[1])
        tz.KeySetValue(tz.KeyAdd(t)[0], pos[2])

        # rotation
        rx.KeySetValue(rx.KeyAdd(t)[0], rot[0])
        ry.KeySetValue(ry.KeyAdd(t)[0], rot[1])
        rz.KeySetValue(rz.KeyAdd(t)[0], rot[2])

    exporter = fbx.FbxExporter.Create(manager, "")
    exporter.Initialize(export_path, -1, manager.GetIOSettings())
    exporter.Export(scene)
    exporter.Destroy()

    print("✔ FBX saved to", export_path)

def export_to_exr(
    prediction: Prediction,
    export_dir: str,
    orig_hw =[1080,1920]
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
        depth = cv2.resize(depth,(orig_hw[1],orig_hw[0]))
        d = depth[...,0]
        d = (d - d.min()) / (d.max() - d.min())
        # if args.norm:
        depth[...,-1] = d
        mvwrite(depth_save_path,depth,precision='half')
        image_vis = images_u8[idx]
        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.resize(image_vis,(orig_hw[1],orig_hw[0]))
        imageio.imwrite(image_save_path, image_vis)
    #fbx output
    cameras_dir = os.path.join(export_dir,f'from{start_idx_img}to{start_idx_img+prediction.depth.shape[0]}.fbx')
    # export_cameras_to_fbx(prediction, export_dir)
    export_camera_anim_fbx(prediction, cameras_dir)

