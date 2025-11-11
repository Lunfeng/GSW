import os
import random
import json

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussian: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0]):
        print("scene init")
        # model_path: data/output
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussian
        # load_iteration 设置迭代次数，根据模型后缀决定
        if load_iteration:
            if load_iteration == -1:
                # 找到模型名称
                # render脚本使用
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # source_path: data
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.message_length)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.message_length)
        else:
            assert False, "Could not recognize scene type!"
        if not self.loaded_iter:
            # ply_path: points3D.ply  model_path: /data/output
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                # 将Point3D文件写入input.ply文件
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            # 在eval环境中会有test_cameras
            # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            #                       image_path=image_path, image_name=image_name, width=width, height=height)
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        # shuffle：让列表中的元素随机排序
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # cameras_extents:所有相机距离相机中点的最远距离
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # train 1.0
        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))