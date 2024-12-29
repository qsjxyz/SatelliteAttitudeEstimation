import torch_bingham
import torch
import torch.nn as nn
from pytorch3d import transforms
from torch.nn import functional as F
from lib.utils.register import Registers

class BinghamAddLogLoss(nn.Module):
    def __init__(self, nm, epsilon=0.95, use_l1=False):
        super(BinghamAddLogLoss, self).__init__()
        self.nm = nm
        self.epsilon = epsilon
        self.use_l1 = use_l1

    def forward(self, data):
        pred_q, Zbatch, weights, gt_q = data[0], data[1], data[2], data[3]
        Zbatch = Zbatch.reshape(-1, 3)
        pred_q = pred_q.reshape(-1, 4)
        gt_q = gt_q.reshape(-1, 1, 4).repeat([1, self.nm, 1]).reshape(-1, 4)

        pred_q = transforms.standardize_quaternion(pred_q)
        gt_q = transforms.standardize_quaternion(gt_q)

        p = torch_bingham.bingham_prob(pred_q, Zbatch, gt_q)

        if self.nm != 1:
            p = p.reshape([-1, self.nm])

            if self.use_l1:
                l1 = torch.abs(pred_q - gt_q).sum(-1).reshape(-1, self.nm)
                best_indices = l1.argmin(1)
            else:
                best_indices = p.argmax(1)

            all_assignments = - torch.mean(p)
            best_assignment = - torch.mean(p[torch.arange(p.shape[0]), best_indices])

            rwta_loss = (self.epsilon - 1 / self.nm) * best_assignment + (
                    1 - self.epsilon) * 1 / self.nm * all_assignments

            weights = F.softmax(weights, dim=-1)
            mb_loss = -torch.mean(torch.logsumexp(torch.log(weights) + p, dim=-1))
            bingham_loss = rwta_loss + mb_loss
        else:
            bingham_loss = - torch.mean(p)

        bs, _ = weights.size()
        how_max, which_max = torch.max(weights, 1)
        pred_q = data[0][torch.arange(bs), which_max].squeeze()
        pred_q = transforms.standardize_quaternion(pred_q)
        truth_q = transforms.standardize_quaternion(data[3])
        l1_loss = torch.mean(torch.norm((pred_q - truth_q), dim=1, p=1), dim=0)
        log_loss = torch.log2(2*l1_loss+1)

        return bingham_loss + log_loss*3

@Registers.LOSSES.register
def binghamAddLogLoss(extra_parameter=None):
    nm = extra_parameter['nm']
    return BinghamAddLogLoss(nm)