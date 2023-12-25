import numpy as np
import torch
from torch import nn


class PositionEncodings1D(object):
    """Positional encodings for `1D` sequences.

    Implements the following equations:

    PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
    PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_model})

    Where d_model is the number of positional features. Also known as the
    depth of the positional encodings. These are the positional encodings
    proposed in [2].
    """
    
    def __init__(self, num_pos_feats, temperature, alpha):
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.alpha = alpha

    def __call__(self, seq_length):
        angle_rads = self.get_angles(
            np.arange(seq_length)[:, np.newaxis],
            np.arange(self.num_pos_feats)[np.newaxis, :]
        )

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        pos_encoding = pos_encoding.astype(np.float32)

        return torch.from_numpy(pos_encoding)

    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(
            self.temperature, (2 * (i//2)) / np.float32(self.num_pos_feats))
        return self.alpha * pos * angle_rates