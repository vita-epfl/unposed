# model.py
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, pose_dim=96, h_dim=32, num_layers=1, dropout=0.2, dev='cuda'):
        super(Encoder, self).__init__()
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
        # x is in shape of (batch, seq_len, feature_dim)
        batch, seq_len, l = x.shape
        state_tuple = (torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev),
                       torch.zeros(self.num_layers, batch, self.h_dim, device=self.dev))
        x = x.contiguous()
        _, state_tuple = self.encoder(x, state_tuple)

        last_frame = x[:, -1, :]  # dim: (batch, pose_dim)
        state_tuple = state_tuple[0][-1, :, :].unsqueeze(0), state_tuple[1][-1, :, :].unsqueeze(0)
        return last_frame, state_tuple


class Decoder(nn.Module):
    def __init__(self, pose_dim=16, h_dim=32, num_layers=1, dropout=0.2, seq_len=25, dev='cuda'):
        super(Decoder, self).__init__()
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
    def __init__(self, pose_dim=96, h_dim=64, num_layers=1, dropout=0.2, seq_len=25, dev='cuda'):
        super(LstmAutoEncoder, self).__init__()

        self.encoder = Encoder(pose_dim, h_dim, num_layers, dropout, dev)
        self.decoder = Decoder(pose_dim, h_dim, 1, dropout, seq_len, dev)

    def forward(self, x):
        last_output, decoder_init_state = self.encoder(x)
        return self.decoder(last_output, decoder_init_state)

class Cl(nn.Module):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    Partially ported from: https://github.com/XifengGuo/DCEC/ and https://github.com/michaal94/torch_DCEC/
    # Example
    ```
        cl = ClusteringLayer(n_clusters=10)
    ```
    # Arguments
        n_clusters: number of clusters.
        input_dim: size of input data with shape `(n_samples, n_features)`
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, input_dim, weights=None, alpha=1.0, **kwargs):
        super(Cl, self).__init__()
        if weights is not None:
            assert weights.shape[1] == input_dim
        self.n_clusters = n_clusters
        self.input_dim = input_dim  # (n_samples, n_features)
        self.alpha = alpha
        self.initial_weights = weights
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, int(input_dim)))
        self.clusters = nn.init.xavier_uniform_(self.clusters)
        if self.initial_weights is not None:
            self.initial_weights = torch.from_numpy(self.initial_weights)
            self.clusters = nn.Parameter(self.initial_weights)
            del self.initial_weights
        self.input_dim = self.clusters.size(1)

    def forward(self, x):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            x: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q_denom = (x.unsqueeze(1) - self.clusters) ** 2
        q_denom = q_denom.sum(dim=2)
        q_denom /= self.alpha
        q_denom += 1.0
        q = 1.0 / q_denom
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q.t() / q.sum(dim=1)  # Div shapes [20, 1024] / [1024]
        q = q.t()
        return q