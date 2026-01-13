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
import pycolmap
import cv2
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.visualize import visualize_depth
from depth_anything_3.utils.fileutils import mvwrite
import glob
import re
import fbx
from scipy.spatial.transform import Rotation as R
from .glb import _depths_to_world_points_with_colors
TEST = False
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf

def get_next_index(
    folder,
    keyword="jpg",
    scan_dir=False
):
    """
    扫描已有编号，返回下一个编号

    scan_dir = False → 扫描文件 (默认行为)
    scan_dir = True  → 扫描子文件夹
    """

    nums = []

    if scan_dir:
        # ---------- 扫描子文件夹 ----------
        if not os.path.exists(folder):
            return 0

        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isdir(path):
                continue

            m = re.match(r"(\d+)", name)
            if m:
                nums.append(int(m.group(1)))

    else:
        # ---------- 扫描文件 ----------
        files = glob.glob(os.path.join(folder, f"*.{keyword}"))
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
        # cv2_to_fbx = np.diag([1, -1, -1])  # 翻转 Y 和 Z
        cv2_to_fbx = np.diag([1, 1, 1]) 
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

# def export_camera_anim_colmap(
#     prediction,
#     export_dir,
#     state: ColmapWriteState,
#     seq_name="seq"
# ):
#     os.makedirs(export_dir, exist_ok=True)

#     cam_path = os.path.join(export_dir, "cameras.txt")
#     img_path = os.path.join(export_dir, "images.txt")
#     pts_path = os.path.join(export_dir, "points3D.txt")

#     num_frames = prediction.extrinsics.shape[0]

#     # ========= 1. 写 cameras.txt（只写一次） =========
#     if not state.camera_written:
#         K = prediction.intrinsics[0]

#         fx = float(K[0, 0])
#         fy = float(K[1, 1])
#         cx = float(K[0, 2])
#         cy = float(K[1, 2])

#         width = int(round(cx * 2))
#         height = int(round(cy * 2))

#         with open(cam_path, "w") as f:
#             f.write("# Camera list with one line of data per camera:\n")
#             f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
#             f.write("# Number of cameras: 1\n")
#             f.write(
#                 f"1 PINHOLE {width} {height} "
#                 f"{fx} {fy} {cx} {cy}\n"
#             )

#         # points3D.txt 也只初始化一次
#         with open(pts_path, "w") as f:
#             f.write("# 3D point list\n")

#         # images.txt header
#         with open(img_path, "w") as f:
#             f.write("# Image list with two lines per image:\n")
#             f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
#             f.write("# POINTS2D[]\n")

#         state.camera_written = True

#     # ========= 2. 追加写 images.txt =========
#     with open(img_path, "a") as f:
#         for i in range(num_frames):
#             w2c = prediction.extrinsics[i]

#             R_wc = w2c[:3, :3]
#             t_wc = w2c[:3, 3]

#             qx, qy, qz, qw = R.from_matrix(R_wc).as_quat()
#             tx, ty, tz = t_wc

#             image_id = state.next_image_id
#             image_name = f"{seq_name}_frame_{i:06d}.png"

#             f.write(
#                 f"{image_id} {qw} {qx} {qy} {qz} "
#                 f"{tx} {ty} {tz} 1 {image_name}\n\n"
#             )

#             state.next_image_id += 1

#     print(f"✔ appended {num_frames} frames to COLMAP ({seq_name})")

def export_to_exr(
    prediction: Prediction,
    export_dir: str,
    orig_hw =[1080,2048],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
):
    # Use prediction.processed_images, which is already processed image data
    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required but not available")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "depths"), exist_ok=True)
    start_idx_img = get_next_index(os.path.join(export_dir, "images"))
    image_paths = []
    for idx in range(prediction.depth.shape[0]):
        global_idx = idx+start_idx_img
        image_save_path = os.path.join(export_dir, f"images/{global_idx:04d}.jpg")
        image_paths.append(image_save_path)
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
    start_idx_colmap = get_next_index(os.path.join(export_dir, "colmap"),scan_dir=True)
    colmap_export_dir = os.path.join(export_dir,f'colmap',f'from{start_idx_img}to{start_idx_img+prediction.depth.shape[0]}')
    mkdir(colmap_export_dir)
    #Colmap export
    # 1. Data preparation
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )
    num_points = len(points)
    print(f"Exporting to COLMAP with {num_points} points")
    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]
    points_xyf = _create_xyf(num_frames, h, w)
    points_xyf = points_xyf[prediction.conf >= conf_thresh]

    # 2. Set Reconstruction
    reconstruction = pycolmap.Reconstruction()
    if TEST:
        point3d_ids = []
        for vidx in range(num_points):
            point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
            point3d_ids.append(point3d_id)

    for fidx in range(num_frames):
        orig_w, orig_h = orig_hw[1],orig_hw[0]
        if TEST:
            bairitsu = orig_w / 508
            orig_w = int(orig_w/bairitsu)
            orig_h = int(orig_h/bairitsu)
        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h
        elif process_res_method == "crop":
            raise NotImplementedError("COLMAP export for crop method is not implemented")
        else:
            raise ValueError(f"Unknown process_res_method: {process_res_method}")

        pycolmap_intri = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # set point2d and update track
        if TEST:
            point2d_list = []
            points_in_frame = points_xyf[:, 2].astype(np.int32) == fidx
            for vidx in np.where(points_in_frame)[0]:
                point2d = points_xyf[vidx][:2]
                point2d[0] *= orig_w / w
                point2d[1] *= orig_h / h
                point3d_id = point3d_ids[vidx]
                point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
                reconstruction.point3D(point3d_id).track.add_element(
                    image.image_id, len(point2d_list) - 1
                )

        # set and add image
        image.frame_id = image.image_id
        image.name = os.path.basename(image_paths[fidx])
        # image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)

    # 3. Export
    reconstruction.write_text(colmap_export_dir)
