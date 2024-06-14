from torch import nn


class DePOSitLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, model_outputs, input_data):
        return {'loss': model_outputs['loss']}
