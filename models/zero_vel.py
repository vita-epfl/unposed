import torch


class ZeroVel(torch.nn.Module):
    def __init__(self, args):
        super(ZeroVel, self).__init__()
        self.args = args
        self.input_size = self.output_size = int(args.keypoints_num * args.keypoint_dim)

    def forward(self, inputs):
        obs_pose = inputs['observed_pose']
        last_frame = obs_pose[..., -1, :].unsqueeze(-2)
        ndims = len(obs_pose.shape)
        pred_pose = last_frame.repeat([1 for _ in range(ndims - 2)] + [self.args.pred_frames_num, 1])
        outputs = {'pred_pose': pred_pose, 'pred_vel': torch.zeros_like(pred_pose)}

        return outputs
