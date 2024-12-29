import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix
from lib.utils.register import Registers

class ADDLossSingleManifoldMode(nn.Module):
    def __init__(self):
        super(ADDLossSingleManifoldMode, self).__init__()

    def forward(self, data):
        pred, mode, gt, point = data[0], data[1], data[2], data[3]
        num_point = point.shape[1]
        pred = self.manifold2euler(pred, mode)
        point = point.reshape(-1, 3).unsqueeze(1)

        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 3)
        pred = pred.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 3)

        rm_gt = euler_angles_to_matrix(gt, "XYZ")
        point_gt = torch.matmul(point, rm_gt).squeeze()

        rm_pred = euler_angles_to_matrix(pred, "XYZ")
        point_pred = torch.matmul(point, rm_pred).squeeze()

        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

    def manifold2euler(self, manifold, mode):
        m1 = manifold[:, 0]
        m2 = manifold[:, 1]
        m3 = manifold[:, 2]
        m4 = manifold[:, 3]
        mode = torch.where(mode > 0.5, torch.ones(1).cuda(), -torch.ones(1).cuda()).squeeze()
        euler2 = mode * torch.asin(torch.sqrt(m3 ** 2 / (m1 ** 2 + m2 ** 2 + m3 ** 2)))
        euler3 = torch.atan2(m4, m3 / (torch.sin(euler2) + 1e-9))
        tmp = torch.cos(euler2) * torch.cos(euler3)
        euler1 = torch.atan2(m2 / tmp, m1 / tmp)
        euler3 = torch.where(euler3 > 0, euler3, euler3 + 2 * np.pi)
        return torch.stack([euler1, euler2, euler3], dim=-1)

@Registers.LOSSES.register
def addLossSingleManifoldMode(*args, **kwargs):
    return ADDLossSingleManifoldMode()
