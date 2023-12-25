import torch
import torch.nn as nn

from utils.others import pose_from_vel


class DeRPoF(nn.Module):
    def __init__(self, args):
        super(DeRPoF, self).__init__()
        self.args = args
        self.keypoints_num = self.args.keypoints_num
        self.keypoint_dim = self.args.keypoint_dim
        self.features_num = int(args.keypoints_num * args.keypoint_dim)

        # global
        self.global_model = LSTM_g(pose_dim=self.features_num, embedding_dim=args.embedding_dim, h_dim=args.hidden_dim,
                                   dropout=args.dropout)

        # local
        encoder = Encoder(pose_dim=self.features_num, h_dim=args.hidden_dim, latent_dim=args.latent_dim,
                          dropout=args.dropout)
        decoder = Decoder(pose_dim=self.features_num, h_dim=args.hidden_dim, latent_dim=args.latent_dim,
                          dropout=args.dropout)
        self.local_model = VAE(Encoder=encoder, Decoder=decoder)

    def forward(self, inputs):
        pose = inputs['observed_pose']
        # print(pose.dtype)
        vel = (pose[..., 1:, :] - pose[..., :-1, :]).permute(1, 0, 2)
        frames_num, bs, _ = vel.shape

        # global
        global_vel = 0.5 * (vel.view(frames_num, bs, self.keypoints_num, self.keypoint_dim)[:, :, 0]
                            + vel.view(frames_num, bs, self.keypoints_num, self.keypoint_dim)[:, :, 1])

        global_vel = global_vel

        # local
        local_vel = (vel.view(frames_num, bs, self.keypoints_num, self.keypoint_dim)
                     - global_vel.view(frames_num, bs, 1, self.keypoint_dim)).view(frames_num, bs, self.features_num)
        # predict
        global_vel_out = self.global_model(global_vel, self.args.pred_frames_num).view(self.args.pred_frames_num, bs, 1, self.keypoint_dim)
        local_vel_out, mean, log_var = self.local_model(local_vel, self.args.pred_frames_num)
        local_vel_out = local_vel_out.view(self.args.pred_frames_num, bs, self.keypoints_num, self.keypoint_dim)
        # merge local and global velocity
        vel_out = (global_vel_out + local_vel_out)
        pred_vel = vel_out.view(self.args.pred_frames_num, bs, self.features_num).permute(1, 0, 2)
        pred_pose = pose_from_vel(pred_vel, pose[..., -1, :])
        outputs = {'pred_pose': pred_pose, 'pred_vel_global': global_vel_out, 'pred_vel_local': local_vel_out,
                   'mean': mean, 'log_var': log_var}

        return outputs


class LSTM_g(nn.Module):
    def __init__(self, pose_dim, embedding_dim=8, h_dim=16, num_layers=2, dropout=0.1):
        super(LSTM_g, self).__init__()
        self.pose_dim = pose_dim
        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers

        self.embedding_fn = nn.Sequential(nn.Linear(3, embedding_dim), nn.ReLU())
        self.encoder_g = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.decoder_g = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.hidden2g = nn.Sequential(nn.Linear(h_dim, 3))

    def forward(self, global_s, pred_len):
        seq_len, batch, l = global_s.shape
        state_tuple_g = (torch.zeros(self.num_layers, batch, self.h_dim, device=global_s.device, dtype=torch.float32),
                         torch.zeros(self.num_layers, batch, self.h_dim, device=global_s.device, dtype=torch.float32))

        global_s = global_s.contiguous()

        output_g, state_tuple_g = self.encoder_g(
            self.embedding_fn(global_s.view(-1, 3)).view(seq_len, batch, self.embedding_dim), state_tuple_g)

        pred_s_g = torch.tensor([], device=global_s.device)
        last_s_g = global_s[-1].unsqueeze(0)
        for _ in range(pred_len):
            output_g, state_tuple_g = self.decoder_g(
                self.embedding_fn(last_s_g.view(-1, 3)).view(1, batch, self.embedding_dim), state_tuple_g)
            curr_s_g = self.hidden2g(output_g.view(-1, self.h_dim))
            pred_s_g = torch.cat((pred_s_g, curr_s_g.unsqueeze(0)), dim=0)
            last_s_g = curr_s_g.unsqueeze(0)
        return pred_s_g


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, obs_s, pred_len):
        mean, log_var = self.Encoder(obs_s=obs_s)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        preds_s = self.Decoder(obs_s=obs_s, latent=z, pred_len=pred_len)

        return preds_s, mean, log_var


class Encoder(nn.Module):
    def __init__(self, pose_dim, h_dim=32, latent_dim=16, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()

        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC_mean = nn.Linear(h_dim, latent_dim)
        self.FC_var = nn.Linear(h_dim, latent_dim)

    def forward(self, obs_s):
        batch = obs_s.size(1)
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device=obs_s.device, dtype=torch.float32),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=obs_s.device, dtype=torch.float32))
        output, state_tuple = self.encoder(obs_s, state_tuple)
        out = output[-1]
        mean = self.FC_mean(out)
        log_var = self.FC_var(out)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, pose_dim, h_dim=32, latent_dim=16, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.decoder = nn.LSTM(pose_dim, h_dim, num_layers, dropout=dropout)
        self.FC = nn.Sequential(nn.Linear(latent_dim, h_dim))
        self.mlp = nn.Sequential(nn.Linear(h_dim, pose_dim))

    def forward(self, obs_s, latent, pred_len):
        batch = obs_s.size(1)
        decoder_c = torch.zeros(self.num_layers, batch, self.h_dim, device=obs_s.device, dtype=torch.float32)
        last_s = obs_s[-1].unsqueeze(0)
        decoder_h = self.FC(latent).unsqueeze(0)
        decoder_h = decoder_h.repeat(self.num_layers, 1, 1)
        state_tuple = (decoder_h, decoder_c)

        preds_s = torch.tensor([], device=obs_s.device)
        for _ in range(pred_len):
            output, state_tuple = self.decoder(last_s, state_tuple)
            curr_s = self.mlp(output.view(-1, self.h_dim))
            preds_s = torch.cat((preds_s, curr_s.unsqueeze(0)), dim=0)
            last_s = curr_s.unsqueeze(0)

        return preds_s
