import torch.nn as nn
import torch as torch


class LstmEncoder(nn.Module):
    def __init__(self, pose_dim=66, h_dim=256, num_layers=3, dropout=0.2, dev='cuda'):
        super(LstmEncoder, self).__init__()
        self.pose_dim = pose_dim
        self.h_dim = h_dim
        self.dev = dev
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            input_size=self.pose_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,

        ).to(dev)

    def forward(self, x):
        batch, seq_len, l = x.shape
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev))
        x = x.contiguous()
        _, state_tuple = self.encoder(x, state_tuple)

        last_frame = x[:, -1, :]  # dim: (batch, pose_dim)
        state_tuple = state_tuple[0][-1, :, :].unsqueeze(0), state_tuple[1][-1, :, :].unsqueeze(0)
        return last_frame, state_tuple


class LstmDecoder(nn.Module):
    def __init__(self, pose_dim=66, h_dim=256, num_layers=1, dropout=0.2, seq_len=25, dev='cuda'):
        super(LstmDecoder, self).__init__()
        self.pose_dim = pose_dim
        self.seq_len = seq_len
        self.h_dim = h_dim
        self.dev = dev
        self.decoder = nn.LSTM(
            input_size=self.pose_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        ).to(dev)
        self.hidden_to_input_space = nn.Linear(h_dim, pose_dim).to(dev)

    def forward(self, first_input, init_state_tuple):
        state_tuple = init_state_tuple
        batch, _ = first_input.shape
        current_input = first_input.unsqueeze(1)
        pred_s_g = torch.tensor([], device=self.dev)

        for i in range(self.seq_len):
            output, state_tuple = self.decoder(current_input, state_tuple)
            current_input = self.hidden_to_input_space(output.view(-1, self.h_dim))
            current_input = current_input.unsqueeze(1)
            pred_s_g = torch.cat((pred_s_g, current_input), dim=1)

        return pred_s_g  # dim: (batch, seq_len, pos_dim)


class LstmAutoEncoder(nn.Module):
    def __init__(self, pose_dim=66, h_dim=256, num_layers=3, dropout=0.2, seq_len=25, dev='cuda'):
        super(LstmAutoEncoder, self).__init__()

        self.encoder = LstmEncoder(pose_dim, h_dim, num_layers, dropout, dev)
        self.decoder = LstmDecoder(pose_dim, h_dim, 1, dropout, seq_len, dev)
        self.outdim = 3

    def forward(self, x):
        last_output, decoder_init_state = self.encoder(x)
        return self.decoder(last_output, decoder_init_state)

    def encode(self, x):
        _, hidden_state = self.encoder(x)
        cat_hidden_state = torch.cat((hidden_state[0].squeeze(0), hidden_state[1].squeeze(0)), dim=1)
        return cat_hidden_state


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super(EncoderWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        z = self.model.encode(x)
        return z
