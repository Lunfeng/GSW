#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from plyfile import PlyData, PlyElement
from pathlib import Path
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch


class CameraInfo(NamedTuple):
    # 相机ID
    uid: int
    # 旋转矩阵
    R: np.array
    # 平移矩阵
    T: np.array
    # FovY一般指垂直视角场大小
    FovY: np.array
    # FovY一般指水平视角场大小
    FovX: np.array
    image: np.array
    message: torch.Tensor
    mask: torch.Tensor
    image_path: str
    image_name: str
    # 传感器宽度
    width: int
    # 传感器高度
    height: int


# nerf_normalization: Nerf++中的算法,计算所有相机的平均值作为相机中点
# # diagonal是所有相机与计算出来的中点最远的距离
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


# Nerf++中的算法,计算所有相机的平均值作为相机中点
# diagonal是所有相机与计算出来的中点最远的距离
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        # 将数组进行水平方向上的拼接
        # a = np.array([1, 2, 3])
        # b = np.array([4, 5, 6])
        # c = np.array([7, 8, 9])
        #
        # result = np.hstack((a, b, c))
        # # 输出：[1 2 3 4 5 6 7 8 9]
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        # 世界坐标系转相机坐标系
        # 相机坐标系的值是以相机位置作为原点的
        # 相机位置位于世界坐标系中,通过这个关系将相机坐标系转到世界坐标系
        W2C = getWorld2View2(cam.R, cam.T)
        # 相机坐标系转师姐坐标系
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    # center: 所有相机的平均值中点
    # diagnoal: 所有相机距离中点的最远距离
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


# cam_extrinsics：图片信息
# cam_intrinsics: 相机信息
# images_folder： image文件夹
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, messages_folder, mask_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        # sys.stdout.write不会输出换行符
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()
        # image.bin文件内容
        extr = cam_extrinsics[key]
        # cameras.bin 文件内容
        intr = cam_intrinsics[extr.camera_id]
        # 传感器宽度
        height = intr.height
        # 传感器高度
        width = intr.width
        # 相机id
        uid = intr.id
        # 四元数转化为旋转矩阵
        R = np.transpose(qvec2rotmat(extr.qvec))
        # 平移矩阵
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            # 视场角
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        message_path = os.path.join(messages_folder, f"{image_name}.pt")
        message = torch.load(message_path) if os.path.exists(message_path) else None

        mask_path = os.path.join(mask_folder, f'{image_name}.png')
        mask = torch.from_numpy(np.array(Image.open(mask_path))) if os.path.exists(mask_path) else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, message=message, mask=mask,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # 数组拼接，将两个数组合并为同一个
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# 读取场景信息
# path： data文件夹
def readColmapSceneInfo(path, images, eval, message_length, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 如果没有image则创建
    reading_dir = "images" if images == None else images
    message_dir = f"messages_{message_length}"
    mask_dir = "masks"
    # 读取camera信息
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           messages_folder=os.path.join(path, message_dir),
                                           mask_folder=os.path.join(path, mask_dir)
                                           )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # 如果eval，则每隔llffhold个元素取一个相机，否则就取全部相机数据
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    # 使用Nerf++中的算法,计算出所有相机的平均值作为相机中点
    # 和所有相机与计算出来的中点最远的距离
    # 返回值translate: -center(为什么是负的没看懂)
    # 返回值radius:所有相机与计算出来的中点最远的距离 * 1.1
    # nerf_normalization {'translate': array([-0.14458875, -0.03369861, -0.02938098], dtype=float32), 'radius': 5.557050561904908}
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            # 读取point3D文件的内容
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    # pcd结构包括位置position  颜色color和法向量normal
    # position.shape: [[x, y, z],[x, y, z]]
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, message_length, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            message_dir = f"messages_{message_length}"
            message_folder = os.path.join(path, message_dir, f"{image_name}.pt")
            message = torch.load(message_folder) if os.path.exists(message_folder) else None

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, message=message, mask=None,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, message_length, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, message_length, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        # train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
