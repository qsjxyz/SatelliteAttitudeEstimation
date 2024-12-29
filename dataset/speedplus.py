import torch.utils.data as data
from PIL import Image
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from lib.utils.fps import farthest_point_sample
from lib.utils.register import Registers
from pytorch3d import transforms as transforms3d

@Registers.DATASETS.register
class SPEEDReal(data.Dataset):
    def __init__(self, mode, imgSize, root, objlist=None):
        self.objlist = ['TANGO']
        self.imgSize = imgSize
        self.obj2id = {'TANGO': 0}
        self.mode = mode

        self.list_rgb = []
        self.list_label = []
        self.root = root

        assert (self.mode in ['train', 'val', 'test']), 'wrong dataset type'

        if self.mode in ['train', 'val']:
            labelDir = os.path.join(self.root, 'synthetic', self.mode+'.txt')
            self.root = os.path.join(self.root, 'synthetic')
        else:
            labelDir = os.path.join(self.root, 'sunlamp', self.mode+'.txt')
            self.root = os.path.join(self.root, 'sunlamp')
        input_file = os.path.join(labelDir)
        inputlines = open(input_file).readlines()
        for inputlinetmp in inputlines:
            inputline = inputlinetmp.strip('\n').split(' ')
            self.list_rgb.append(os.path.join(self.root, 'images', inputline[0]))
            self.list_label.append([float(inputline[1]), float(inputline[2]), float(inputline[3]), float(inputline[4])])

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

        quat = self.list_label[index]

        meta = self.list_label[index]
        output['meta'] = torch.from_numpy((np.array(meta) / 180. * np.pi).astype(np.float32))

        quat = torch.from_numpy(np.array(quat).astype(np.float32))
        output['quat'] = quat
        RM = transforms3d.quaternion_to_matrix(quat)
        output['RM'] = RM
        r6d = transforms3d.matrix_to_rotation_6d(RM)
        output['r6d'] = r6d

        # reutrn 图片, 标签(欧拉角形式)，点云，标签（四元数）
        return output

    def __len__(self):
        return self.length
