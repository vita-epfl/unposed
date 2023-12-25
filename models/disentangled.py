import torch
import torch.nn as nn

from models.pv_lstm import PVLSTM
from models.zero_vel import ZeroVel
from utils.others import pose_from_vel


class Disentangled(nn.Module):
    def __init__(self, args):
        super(Disentangled, self).__init__()
        self.args = args

        # global
        global_args = args.copy()
        global_args.keypoints_num = 1
        self.global_model = ZeroVel(global_args)

        # local
        local_args = args.copy()
        local_args.keypoints_num = args.keypoints_num - global_args.keypoints_num
        self.local_model = PVLSTM(local_args)

    def forward(self, inputs):
        pose = inputs['observed_pose']

        # global
        global_pose = pose[..., : self.args.keypoint_dim]
        global_inputs = {'observed_pose': global_pose}

        # local
        repeat = torch.ones(len(global_pose.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        local_pose = pose[..., self.args.keypoint_dim:] - global_pose.repeat(tuple(repeat))
        local_inputs = {'observed_pose': local_pose}

        # predict
        global_outputs = self.global_model(global_inputs)
        local_outputs = self.local_model(local_inputs)

        # merge local and global velocity
        global_vel_out = global_outputs['pred_vel']
        local_vel_out = local_outputs['pred_vel']
        repeat = torch.ones(len(global_vel_out.shape), dtype=int)
        repeat[-1] = self.local_model.args.keypoints_num
        pred_vel = torch.cat((global_vel_out, local_vel_out + global_vel_out.repeat(tuple(repeat))), dim=-1)
        pred_pose = pose_from_vel(pred_vel, pose[..., -1, :])
        outputs = {'pred_pose': pred_pose, 'pred_vel': pred_vel}

        return outputs
