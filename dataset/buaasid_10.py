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

@Registers.DATASETS.register
class BUAASID_10(data.Dataset):
    def __init__(self, mode, imgSize, root, objlist=None):
        if objlist is None:
            objlist = ['a2100', 'cobe', 'early-bird', 'fengyun', 'galileo']
        self.objlist = objlist
        self.obj2id = {'a2100': 0, 'cobe': 1, 'early-bird': 2, 'fengyun': 3, 'galileo': 4}

        self.mode = mode
        assert (self.mode in ['train', 'val', 'test']), 'wrong dataset type'
        self.imgSize = imgSize
        self.list_rgb = []
        self.list_item = []
        self.list_point = []
        self.list_label = []
        self.list_keypoint = []
        self.ori_model = {}
        self.list_confidenceImg = []
        self.keypoint_pnp = {}
        self.root = root

        for id, item in enumerate(self.objlist):
            keypoint_file = open(os.path.join(self.root, 'BUAA-SID', item, 'keyPointFPS', item + '-00001.txt'))
            keypoint_lines = keypoint_file.readlines()
            keypoint = []
            for keypoint_line in keypoint_lines:
                keypoint_line = keypoint_line.strip('\n').split(' ')
                keypoint.append([float(keypoint_line[0]), float(keypoint_line[1]), float(keypoint_line[2])])
            self.keypoint_pnp[item] = keypoint

            ori_model_file = open(os.path.join(self.root, '3d', item+'.off'))
            ori_model_lines = ori_model_file.readlines()
            ori_model_point = []
            for ori_model_point_line in ori_model_lines[2:]:
                ori_model_point_line = ori_model_point_line.strip('\n').split(' ')
                ori_model_point.append([float(ori_model_point_line[0]), float(ori_model_point_line[1]), float(ori_model_point_line[2])])
            ori_model_point = np.array(ori_model_point)
            ori_model_point = farthest_point_sample(ori_model_point, 1024)
            ori_model_point = self._pointNormalize(ori_model_point)
            self.ori_model[item] = ori_model_point

        for id, item in enumerate(self.objlist):
            input_file = os.path.join(self.root, 'label', '1-1-8', '01', item, '{0}.txt'.format(self.mode))
            inputlines = open(input_file).readlines()
            for inputlinetmp in inputlines:
                inputline = inputlinetmp.strip('\n').split(' ')
                self.list_rgb.append(os.path.join(self.root, 'BUAA-SID', item, 'image', inputline[0]))
                self.list_item.append(item)
                self.list_point.append(os.path.join(self.root, 'BUAA-SID', item, 'pointCloud-256', inputline[0].split('.')[0]+'.txt'))
                self.list_keypoint.append(os.path.join(self.root, 'BUAA-SID', item, 'keyPointFPS', inputline[0].split('.')[0] + '.txt'))
                self.list_label.append([-float(inputline[1]), -float(inputline[2]), -float(inputline[3])])
                self.list_confidenceImg.append(os.path.join(self.root, 'confidenceImg', item, inputline[0].split('.')[0]+'.pth'))

        self.length = len(self.list_rgb)
        self.guass1D = cv2.getGaussianKernel(ksize=5, sigma=1)
        self.guass2D = (self.guass1D * self.guass1D.T)/(self.guass1D.max()**2)

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

        point_file = open(self.list_point[index])
        point_lines = point_file.readlines()
        point = []
        for point_line in point_lines:
            point_line = point_line.strip('\n').split(' ')
            point.append([float(point_line[0]), float(point_line[1]), float(point_line[2])])
        point = np.array(point)
        output['point'] = torch.from_numpy((point/255.).astype(np.float32))

        pointImg = self._point2Img(point)
        tfPoint = transforms.Compose([  # 常用的数据变换器
            transforms.Resize((self.imgSize, self.imgSize)),
            transforms.ToTensor()
        ])
        pointImg = tfPoint(pointImg)
        pointImg = pointImg / 255.
        output['pointImg'] = pointImg

        keypoint_file = open(self.list_keypoint[index])
        keypoint_lines = keypoint_file.readlines()
        keypoint = []
        for keypoint_line in keypoint_lines:
            keypoint_line = keypoint_line.strip('\n').split(' ')
            keypoint.append([float(keypoint_line[0]), float(keypoint_line[1]), float(keypoint_line[2])])
        output['keypoint'] = torch.from_numpy(np.array(keypoint).astype(np.float32))

        keypointImg = self._getKeypointImg(keypoint)
        output['keypointImg'] = keypointImg

        imgWithPoint = torch.cat([img, pointImg], dim=0)
        output['imgWithPoint'] = imgWithPoint

        meta = self.list_label[index]
        output['meta'] = torch.from_numpy((np.array(meta)/180.*np.pi).astype(np.float32))

        RM = self.euler2RM(meta)
        RMTensor = torch.from_numpy(np.array(RM).astype(np.float32))
        output['RM'] = RMTensor

        keypoint_pnp = self.keypoint_pnp[self.list_item[index]]
        output['keypoint_pnp'] = torch.from_numpy(np.array(keypoint_pnp).astype(np.float32))

        ori_model = self.ori_model[self.list_item[index]]
        output['ori_model'] = torch.from_numpy(np.array(ori_model).astype(np.float32))

        quat = transforms3d.matrix_to_quaternion(RMTensor)
        output['quat'] = quat

        r6d = transforms3d.matrix_to_rotation_6d(RMTensor)
        output['r6d'] = r6d

        output['class'] = self.list_item[index]
        output['class_id'] = self.obj2id[output['class']]

        # RM1 = torch.tensor(RM)
        # quat1 = transforms3d.matrix_to_quaternion(RM1)
        # # solve = ops3d.efficient_pnp((torch.tensor(keypoint_pnp)).unsqueeze(0),
        # #                             (torch.tensor(keypoint)[:, :2]).unsqueeze(0),
        # #                             skip_quadratic_eq=True)
        # # quat2 = transforms3d.matrix_to_quaternion(solve.R)
        # point3s = np.array(keypoint_pnp).astype(np.double) / 2560
        # point2s = np.array(keypoint).astype(np.double)[:, :2] + 128
        #
        # camera = np.array(([2560, 0, 128],
        #                    [0, 2560, 128],
        #                    [0, 0, 1]), dtype=np.float)
        # # dist=dist.T
        # dist = np.zeros((5, 1), dtype=np.float)
        # _, r, t, _ = cv2.solvePnPRansac(point3s, point2s, camera, dist,
        #                                 flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, confidence=0.99)  # 计算雷达相机外参,r-旋转向量，t-平移向量
        # R = cv2.Rodrigues(r)[0]  # 旋转向量转旋转矩阵
        # RM2 = torch.tensor(R)
        # quat2 = transforms3d.matrix_to_quaternion(RM2)
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
