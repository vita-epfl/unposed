import torch.nn as nn


class MSEPose(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, model_outputs, input_data):
        pose_loss = self.mse(model_outputs['pred_pose'], input_data['future_pose'])

        loss = pose_loss
        outputs = {'pose_loss': pose_loss}

        outputs['loss'] = loss

        return outputs
