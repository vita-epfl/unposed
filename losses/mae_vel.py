import torch
import torch.nn as nn


class MAEVel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mae = nn.L1Loss()
        self.bce = nn.BCELoss()

    def forward(self, model_outputs, input_data):
        observed_pose = input_data['observed_pose']
        future_pose = input_data['future_pose']
        future_vel = torch.cat(((future_pose[..., 0, :] - observed_pose[..., -1, :]).unsqueeze(-2),
                                future_pose[..., 1:, :] - future_pose[..., :-1, :]), -2)
        vel_loss = self.mae(model_outputs['pred_vel'], future_vel)

        loss = vel_loss
        outputs = {'vel_loss': vel_loss}

        outputs['loss'] = loss

        return outputs
