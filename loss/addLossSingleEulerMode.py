import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix, quaternion_apply
from lib.utils.register import Registers

class ADDLossSingleEulerMode(nn.Module):
    def __init__(self):
        super(ADDLossSingleEulerMode, self).__init__()

    def forward(self, data):
        pred, gt, point = data[0], data[1], data[2]
        num_point = point.shape[1]
        point = point.reshape(-1, 3).unsqueeze(1)

        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 3)
        pred = pred.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 3)

        rm_gt = euler_angles_to_matrix(gt, "XYZ")
        point_gt = torch.matmul(point, rm_gt).squeeze()

        rm_pred = euler_angles_to_matrix(pred, "XYZ")
        point_pred = torch.matmul(point, rm_pred).squeeze()

        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

@Registers.LOSSES.register
def addLossSingleEulerMode(*args, **kwargs):
    return ADDLossSingleEulerMode()