
import numpy as np
import torch
import torch.nn as nn
from .pua_loss import PUALoss
from models.pgbig.data_proc import Preprocess, Human36m_Preprocess, AMASS_3DPW_Preprocess


def smooth(src, sample_len, kernel_size):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].clone()
    smooth_data = src_data.clone()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = torch.mean(src_data[:, kernel_size:i+1], dim=1)
    return smooth_data


class PGBIG_PUALoss(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

        if args.inner_type == "PUAL":
            if 'S' in args.tasks:
                self.pual1 = PUALoss(args).to(args.device)
                self.pual2 = PUALoss(args).to(args.device)
                self.pual3 = PUALoss(args).to(args.device)
                self.pual4 = PUALoss(args).to(args.device)
            else:
                self.pual = PUALoss(args).to(args.device)

        if args.pre_post_process == 'human3.6m':
            self.preprocess = Human36m_Preprocess(args).to(args.device)
        elif args.pre_post_process == 'AMASS' or args.pre_post_process == '3DPW':
            self.preprocess = AMASS_3DPW_Preprocess(args).to(args.device)
        else:
            self.preprocess = Preprocess(args).to(args.device)

        for p in self.preprocess.parameters():
            p.requires_grad = False


    def forward(self, y_pred, y_true):
        p1 = y_pred['p1']
        p2 = y_pred['p2']
        p3 = y_pred['p3']
        p4 = y_pred['p4']

        y_future = self.preprocess(y_true['future_pose'], normal=True)
        y_obs = self.preprocess(y_true['observed_pose'], normal=True)

        y_full = torch.cat([y_obs, y_future], dim=1)

        B, T, JC = y_full.shape
        J, C = JC // 3, 3
        _, seq_in, _ = y_obs.shape

        smooth1 = smooth(y_full, sample_len=T, kernel_size=seq_in)
        smooth2 = smooth(smooth1, sample_len=T, kernel_size=seq_in)
        smooth3 = smooth(smooth2, sample_len=T, kernel_size=seq_in)

        # nn.utils.clip_grad_norm_(
        #     list(net_pred.parameters()), max_norm=opt.max_norm)

        if self.args.inner_type == "PUAL":
            if 'S' in self.args.tasks:
                loss_p3d_4 = self.pual4({'pred_pose': p4}, {'future_pose': y_full})['loss']
                loss_p3d_3 = self.pual3({'pred_pose': p3}, {'future_pose': smooth1})['loss']
                loss_p3d_2 = self.pual2({'pred_pose': p2}, {'future_pose': smooth2})['loss']
                loss_p3d_1 = self.pual1({'pred_pose': p1}, {'future_pose': smooth3})['loss']
            else:
                loss_p3d_4 = self.pual({'pred_pose': p4}, {'future_pose': y_full})['loss']
                loss_p3d_3 = self.pual({'pred_pose': p3}, {'future_pose': smooth1})['loss']
                loss_p3d_2 = self.pual({'pred_pose': p2}, {'future_pose': smooth2})['loss']
                loss_p3d_1 = self.pual({'pred_pose': p1}, {'future_pose': smooth3})['loss']

        else:
            p3d_sup_4 = y_full.view(B, T, J, C)
            p3d_sup_3 = smooth1.view(B, T, J, C)
            p3d_sup_2 = smooth2.view(B, T, J, C)
            p3d_sup_1 = smooth3.view(B, T, J, C)

            p4 = p4.view(B, T, J, C)
            p3 = p3.view(B, T, J, C)
            p2 = p2.view(B, T, J, C)
            p1 = p1.view(B, T, J, C)

            loss_p3d_4 = torch.mean(torch.norm(p4 - p3d_sup_4, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3 - p3d_sup_3, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p2 - p3d_sup_2, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p1 - p3d_sup_1, dim=3))

        return {
            'loss': (loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1)/4
        }
