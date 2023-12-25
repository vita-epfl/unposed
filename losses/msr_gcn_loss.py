import imp
import torch
import torch.nn as nn
from metrics import ADE
import numpy as np

from models.msr_gcn.utils import data_utils

class Proc(nn.Module):
    def __init__(self, args):
        super(Proc, self).__init__()

        self.args = args

        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        self.dim_used = dimensions_to_use

        self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11], [12], [13], [14, 15, 16], [17], [18], [19, 20, 21]]
        self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
        self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

    def down(self, x, index):
        N, features, seq_len = x.shape
        my_data = x.reshape(N, -1, 3, seq_len)  # x, 22, 3, 10
        da = torch.zeros((N, len(index), 3, seq_len)).to(x.device) # x, 12, 3, 10
        for i in range(len(index)):
            da[:, i, :, :] = torch.mean(my_data[:, index[i], :, :], dim=1)
        da = da.reshape(N, -1, seq_len)
        return da

    def forward(self, x, preproc):
        if preproc:
            x32 = x.permute((0,2,1))
            x22 = x32[:, self.dim_used, :]
            x12 = self.down(x22, self.Index2212)
            x7 = self.down(x12, self.Index127)
            x4 = self.down(x7, self.Index74)

            return {
                "p32":x32,
                "p22":x22,
                "p12":x12,
                "p7":x7,
                "p4":x4
            }
        else:
            return x

def L2NormLoss_train(gt, out):
    '''
    ### (batch size,feature dim, seq len)
    等同于 mpjpe_error_p3d()
    '''

    batch_size, _, seq_len = gt.shape
    gt = gt.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
    out = out.view(batch_size, -1, 3, seq_len).permute(0, 3, 1, 2).contiguous()
    loss = torch.mean(torch.norm(gt - out, 2, dim=-1))
    return loss

class MSRGCNLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.proc = Proc(args)
        self.args = args

    def forward(self, model_outputs, input_data):
        gt = torch.cat([input_data['observed_pose'].clone(), input_data['future_pose'].clone()], dim=1)
        output_size = gt.shape[1]
        gt = gt.reshape((gt.shape[0], gt.shape[1], -1))
        gt = self.proc(gt, True) # batch_size * (66|36|21|12) * T
        out = {
            "p22":model_outputs["p22"], # batch_size * (66|36|21|12) * T
            "p12":model_outputs["p12"],
            "p7":model_outputs["p7"],
            "p4":model_outputs["p4"]
        }
        losses = {}
        for k in out.keys():
            losses[k] = 0
        # frames = [i for i in [11,13,17,19,23,34] if i < output_size]
        
        for k in out.keys():
            temp = out[k]
            # if "22" in k:
            #     batch_size, _, seq_len = gt[k].shape
                # for frame in frames:
                #     losses[frame]=torch.mean(torch.norm(gt[k].view(batch_size,-1,3,seq_len)[:,:,:,frame+10-1]- \
                #                                         temp.view(batch_size, -1, 3, seq_len)[:,:,:,frame+10-1], 2, -1))
            losses[k] += L2NormLoss_train(gt[k], temp)
        
        final_loss = 0
        for k in out.keys():
            final_loss+= losses[k]

        return {'loss': final_loss}