import torch.utils.data as data
from PIL import Image
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from lib.utils.fps import farthest_point_sample
from lib.utils.register import Registers
from pytorch3d import transforms as transforms3d
from pytorch3d import ops as ops3d
import itertools

@Registers.DATASETS.register
class SPEEDSoftEncode(data.Dataset):
    def __init__(self, mode, imgSize, root, objlist=None):
        self.objlist = ['TANGO']
        self.imgSize = imgSize
        self.obj2id = {'TANGO': 0}
        self.mode = mode

        self.list_rgb = []
        self.list_label = []
        self.list_soft_encode = []
        self.root = root

        assert (self.mode in ['train', 'val', 'test']), 'wrong dataset type'

        min_lim = np.array([-180, -90, 0])
        max_lim = np.array([180, 90, 360])
        nr_bins_per_dim = 24
        d = 3
        nr_total_bins = nr_bins_per_dim ** d
        beta = 6
        delta = beta / nr_bins_per_dim
        var = delta ** 2 / 12

        bins_loc_per_dim = np.linspace(0.0, 1.0, nr_bins_per_dim)
        H_loc_list = list(itertools.product(bins_loc_per_dim, repeat=d))
        H_ori = np.asarray(H_loc_list * (max_lim - min_lim) + min_lim)
        self.H_quat = np.zeros(shape=(nr_total_bins, 4), dtype=np.float32)

        for i in range(nr_total_bins):
            RM = self.euler2RM(H_ori[i])
            RMTensor = torch.from_numpy(np.array(RM).astype(np.float32))
            self.H_quat[i, :] = transforms3d.standardize_quaternion(transforms3d.matrix_to_quaternion(RMTensor)).numpy()

        Boundary_flags = np.logical_or(H_ori[:, 0] == max_lim[0], H_ori[:, 2] == max_lim[2])
        Gymbal_flags = np.logical_and(np.abs(H_ori[:, 1]) == max_lim[1], H_ori[:, 0] != min_lim[0])
        Redundant_flags = np.logical_or(Boundary_flags, Gymbal_flags)

        if self.mode in ['train', 'val']:
            labelDir = os.path.join(self.root, 'synthetic', self.mode + '.txt')
            self.root = os.path.join(self.root, 'synthetic')
        else:
            labelDir = os.path.join(self.root, 'sunlamp', self.mode + '.txt')
            self.root = os.path.join(self.root, 'sunlamp')
        input_file = os.path.join(labelDir)
        inputlines = open(input_file).readlines()
        for inputlinetmp in inputlines:
            inputline = inputlinetmp.strip('\n').split(' ')
            self.list_rgb.append(os.path.join(self.root, 'images', inputline[0]))
            quatList = [float(inputline[1]), float(inputline[2]), float(inputline[3]), float(inputline[4])]
            self.list_label.append(quatList)
            quat = transforms3d.standardize_quaternion(torch.tensor(quatList)).numpy()
            H_prbs = np.exp(
                -2 * (np.arccos(np.minimum(1.0, np.abs(np.sum(quat * self.H_quat, axis=-1)))) / np.pi) ** 2 / var)
            for i in range(nr_total_bins):
                if Redundant_flags[i]:
                    H_prbs[i] = 0
            self.list_soft_encode.append(H_prbs / np.sum(H_prbs))

        print("Object {0} buffer loaded".format(self.mode))

        self.length = len(self.list_rgb)

    def __getitem__(self, index):
        output = {}

        img = self.list_rgb[index]
        tf = transforms.Compose([  # 常用的数据变换器
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((self.imgSize, self.imgSize)),
            transforms.ToTensor()
        ])
        img = tf(img)
        img = img / 255.
        output['img'] = img

        output['ori_histogram_map'] = self.H_quat

        quat = self.list_label[index]

        quat = torch.from_numpy(np.array(quat).astype(np.float32))
        output['quat'] = quat
        RM = transforms3d.quaternion_to_matrix(quat)
        output['RM'] = RM
        meta = transforms3d.matrix_to_euler_angles(RM, "XYZ")
        output['meta'] = meta
        r6d = transforms3d.matrix_to_rotation_6d(RM)
        output['r6d'] = r6d

        output['softEncode'] = torch.from_numpy(self.list_soft_encode[index])

        # reutrn 图片, 标签(欧拉角形式)，点云，标签（四元数）
        return output

    def __len__(self):
        return self.length

    def _point2Img(self, points):
        img = np.zeros([256, 256])
        for point in points:
            img[int(point[0])+128, int(point[1])+128] = 128 - point[2]
        return Image.fromarray(img.astype(np.uint8))

    def _getKeypointImg(self, keypoint):
        img = None
        for i in range(len(keypoint)):
            imgOne = np.zeros([self.imgSize, self.imgSize])
            x = int(max(min(253, keypoint[i][0] + 128), 2) / 256 * self.imgSize)
            y = int(max(min(253, keypoint[i][1] + 128), 2) / 256 * self.imgSize)
            imgOne[x-2:x+3, y-2:y+3] = self.guass2D
            imgOne = np.expand_dims(imgOne, 0)
            if img is None:
                img = imgOne
            else:
                img = np.concatenate((img, imgOne), axis=0)
        return torch.from_numpy(img)

    def _euler2RM(self, euler):
        r = R.from_euler('xyz', euler, degrees=True)
        rotation_matrix = r.as_matrix()
        return rotation_matrix.squeeze()

    def euler2RM(self, euler):
        result = np.array(self._euler2RM([euler[0], 0, 0])) @ \
                 np.array(self._euler2RM([0, euler[1], 0])) @ \
                 np.array(self._euler2RM([0, 0, euler[2]]))
        return result

    def _pointNormalize(self, point):
        centroid = np.mean(point, axis=0)
        point = point-centroid
        m = np.max(np.sqrt(np.sum(point**2, axis=1)))
        point = point / m
        return point

    def _euler2quat(self, pitch, yaw, roll):
        cos_pitch = np.cos(pitch * np.pi / 360)
        sin_pitch = np.sin(pitch * np.pi / 360)
        cos_yaw = np.cos(yaw * np.pi / 360)
        sin_yaw = np.sin(yaw * np.pi / 360)
        cos_roll = np.cos(roll * np.pi / 360)
        sin_roll = np.sin(roll * np.pi / 360)

        q = np.matrix([[sin_yaw * sin_roll * cos_pitch - cos_yaw * cos_roll * sin_pitch],
                       [-sin_yaw * cos_roll * cos_pitch - cos_yaw * sin_roll * sin_pitch],
                       [-cos_yaw * sin_roll * cos_pitch + sin_yaw * cos_roll * sin_pitch],
                       [cos_yaw * cos_roll * cos_pitch + sin_yaw * sin_roll * sin_pitch]])

        return q
