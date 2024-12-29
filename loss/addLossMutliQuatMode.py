import torch
import torch.nn as nn
from pytorch3d.transforms import standardize_quaternion, quaternion_apply
from lib.utils.register import Registers

class ADDLossMultiQuatMode(nn.Module):
    def __init__(self):
        super(ADDLossMultiQuatMode, self).__init__()

    def forward(self, data):
        pred, weights, gt, point = data[0], data[1], data[2], data[3]
        bs, num_point, _ = point.size()
        how_max, which_max = torch.max(weights, 1)
        pred_q = pred[torch.arange(bs), which_max].squeeze()
        pred_q = pred_q.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 4)
        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 4)
        point = point.reshape(-1, 3)
        point_pred = quaternion_apply(pred_q, point)
        point_gt = quaternion_apply(gt, point)
        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

@Registers.LOSSES.register
def addLossMultiQuatMode(*args, **kwargs):
    return ADDLossMultiQuatMode()