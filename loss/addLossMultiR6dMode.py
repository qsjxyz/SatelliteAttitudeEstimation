import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix
from lib.utils.register import Registers

class ADDLossMultiR6dMode(nn.Module):
    def __init__(self):
        super(ADDLossMultiR6dMode, self).__init__()

    def forward(self, data):
        pred, weights, gt, point = data[0], data[1], data[2], data[3]
        bs, num_point, _ = point.size()
        how_max, which_max = torch.max(weights, 1)
        pred = pred[torch.arange(bs), which_max].squeeze()

        point = point.reshape(-1, 3).unsqueeze(1)

        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 6)
        pred = pred.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 6)

        rm_gt = rotation_6d_to_matrix(gt)
        point_gt = torch.matmul(point, rm_gt).squeeze()

        rm_pred = rotation_6d_to_matrix(pred)
        point_pred = torch.matmul(point, rm_pred).squeeze()

        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

@Registers.LOSSES.register
def addLossMultiR6dMode(*args, **kwargs):
    return ADDLossMultiR6dMode()