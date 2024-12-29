import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix, quaternion_invert, standardize_quaternion, matrix_to_quaternion
from lib.utils.register import Registers
import cv2
import numpy as np

class ADDLossSoftEncode(nn.Module):
    def __init__(self):
        super(ADDLossSoftEncode, self).__init__()

    def forward(self, data):
        softEncodePred, oriHistogramMap, gt, point = data[0], data[1], data[2], data[3]
        bs, num_point, _ = point.size()

        exps = torch.exp(softEncodePred-softEncodePred.max(dim=-1, keepdim=True)[0])
        oriPmf = exps/exps.sum(dim=-1, keepdim=True)
        q_est = [self.quat_weighted_avg(oriHistogramMap[i, :].cpu().numpy(), oriPmf[i, :].cpu().numpy()) for i in range(bs)]
        pred = torch.tensor(q_est).cuda()

        point = point.reshape(-1, 3).unsqueeze(1)

        gt = gt.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 4)
        pred = pred.unsqueeze(1).repeat(1, num_point, 1).reshape(-1, 4)

        rm_gt = quaternion_to_matrix(gt)
        point_gt = torch.matmul(point, rm_gt).squeeze()

        rm_pred = quaternion_to_matrix(pred)
        point_pred = torch.matmul(point, rm_pred).squeeze()

        return torch.mean(torch.norm((point_pred - point_gt), dim=1), dim=0)

    def quat_weighted_avg(self, Q, W):
        N = np.size(Q, 0)

        # Compute A TODO: vectorize
        A = np.zeros(shape=(4, 4), dtype=np.float32)
        for i in range(N):
            a = np.matrix([Q[i, 0], Q[i, 1], Q[i, 2], Q[i, 3]])
            A += a.transpose() * a * W[i]

        s, v = np.linalg.eig(A)
        idx = np.argsort(s)

        q_avg = v[:, idx[-1]]  # 0

        # Due to numerical errors, we need to enforce normalization
        q_avg = q_avg / np.linalg.norm(q_avg)

        return q_avg

@Registers.LOSSES.register
def addLossSoftEncode(*args, **kwargs):
    return ADDLossSoftEncode()