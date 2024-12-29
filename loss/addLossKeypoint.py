import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert, standardize_quaternion, matrix_to_quaternion
from lib.utils.register import Registers
import cv2
import numpy as np

class ADDLossKeypoint(nn.Module):
    def __init__(self):
        super(ADDLossKeypoint, self).__init__()

    def forward(self, data):
        keypointPred, classPred, keypointPnP, gt, point = data[0], data[1], data[2], data[3], data[4]
        bs, num_point, _ = point.size()

        classPred = classPred.argmax(dim=-1)
        keypointPred = keypointPred[torch.arange(bs), classPred, :, :].squeeze()
        predR = []
        for i in range(bs):
            point3s = keypointPnP[i, :, :].cpu().detach().numpy() / 2560
            point2s = keypointPred[i, :, :].cpu().detach().numpy() + 128

            camera = np.array(([2560, 0, 128],
                               [0, 2560, 128],
                               [0, 0, 1]), dtype=np.double)
            # dist=dist.T
            dist = np.zeros((5, 1))
            _, r, t, _ = cv2.solvePnPRansac(point3s, point2s, camera, dist,
                                            flags=cv2.SOLVEPNP_EPNP)  # 计算雷达相机外参,r-旋转向量，t-平移向量
            R = cv2.Rodrigues(r)[0]  # 旋转向量转旋转矩阵
            predR.append(R)

        predR = torch.from_numpy(np.array(predR, dtype=np.float32)).cuda()
        pred = standardize_quaternion(quaternion_invert(matrix_to_quaternion(predR)))

        point = point.reshape(-1, 3).unsqueeze(1)
        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 4)
        pred = pred.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 4)

        rm_gt = quaternion_to_matrix(gt)
        point_gt = torch.matmul(point, rm_gt).squeeze()

        rm_pred = quaternion_to_matrix(pred)
        point_pred = torch.matmul(point, rm_pred).squeeze()

        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

@Registers.LOSSES.register
def addLossKeypoint(*args, **kwargs):
    return ADDLossKeypoint()