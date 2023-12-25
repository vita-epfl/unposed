import numpy as np
import torch
import torch.nn as nn

from models.sts_gcn.utils import data_utils


class MPJPE(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        

    def forward(self, y_pred, y_true):
        
        y_pred = y_pred['pred_pose'] # B,T,JC
        y_true = y_true['future_pose'] # B,T,JC

        B,T,JC = y_pred.shape
        assert JC % self.args.nJ == 0, "Number of joints * dim of each joint is not dividable by nJ"
        J = self.args.nJ
        C = JC // J

        y_pred = y_pred.view(B, T, J, C)
        y_true = y_true.view(B, T, J, C)

        l = torch.norm(y_pred - y_true, dim=-1) # B,T,J
        l = torch.mean(l)

        return {
          'loss' : l
        }
