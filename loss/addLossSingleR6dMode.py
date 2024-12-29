import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, quaternion_apply
from lib.utils.register import Registers

class ADDLossSingleR6dMode(nn.Module):
    def __init__(self):
        super(ADDLossSingleR6dMode, self).__init__()

    def forward(self, data):
        pred, gt, point = data[0], data[1], data[2]
        num_point = point.shape[1]
        point = point.reshape(-1, 3).unsqueeze(1)

        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 6)
        pred = pred.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 6)

        rm_gt = rotation_6d_to_matrix(gt)
        point_gt = torch.matmul(point, rm_gt).squeeze()

        rm_pred = rotation_6d_to_matrix(pred)
        point_pred = torch.matmul(point, rm_pred).squeeze()

        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

@Registers.LOSSES.register
def addLossSingleR6dMode(*args, **kwargs):
    return ADDLossSingleR6dMode()