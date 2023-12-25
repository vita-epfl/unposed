import numpy as np
from torch import nn
import torch
from .utils import data_utils


def get_dct_matrix(N, device):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return torch.FloatTensor(dct_m).to(device), torch.FloatTensor(idct_m).to(device)


def dct_transform_torch(data, dct_m, dct_n):
    '''
    B, 60, 35
    '''
    batch_size, features, seq_len = data.shape

    data = data.contiguous().view(-1, seq_len)  # [180077*60， 35]
    data = data.permute(1, 0)  # [35, b*60]

    out_data = torch.matmul(dct_m[:dct_n, :], data)  # [dct_n, 180077*60]
    out_data = out_data.permute(1, 0).contiguous().view(-1, features, dct_n)  # [b, 60, dct_n]
    return out_data

def reverse_dct_torch(dct_data, idct_m, seq_len):
    '''
    B, 60, 35
    '''
    batch_size, features, dct_n = dct_data.shape

    dct_data = dct_data.permute(2, 0, 1).contiguous().view(dct_n, -1)  # dct_n, B*60
    out_data = torch.matmul(idct_m[:, :dct_n], dct_data).contiguous().view(seq_len, batch_size, -1).permute(1, 2, 0)
    return out_data


class Proc(nn.Module):
    def __init__(self, args):
        super(Proc, self).__init__()

        self.dct_used = args.dct_used
        self.input_n = args.input_n
        self.output_n = args.output_n
        self.dct_m, self.idct_m = get_dct_matrix(self.input_n + self.output_n, args.device)
        self.global_min = args.global_min
        self.global_max = args.global_max

        self.args = args

        self.dim_repeat_22 = [27, 28, 29, 27, 28, 29, 42, 43, 44, 48, 49, 50, 57, 58, 59, 63, 64, 65]
        self.dim_repeat_32 = [48,49,50, 72,73,74, 60,61,62, 69,70,71, 84,85,86, 93,94,95]

        self.dim_replace = [0,1,2, 3,4,5, 18,19,20, 33,34,35]

        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        self.dim_used = dimensions_to_use

        self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11], [12], [13], [14, 15, 16], [17], [18], [19, 20, 21]]
        self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
        self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

    def down(self, x, index):
        N, features, seq_len = x.shape
        my_data = x.reshape(N, -1, 3, seq_len)  # x, 22, 3, 10
        da = torch.zeros((N, len(index), 3, seq_len)).to(x.device) # x, 12, 3, 10
        for i in range(len(index)):
            da[:, i, :, :] = torch.mean(my_data[:, index[i], :, :], dim=1)
        da = da.reshape(N, -1, seq_len)
        return da

    def forward(self, x, preproc):
        if preproc:
            x32 = x.permute((0,2,1))
            x32 = torch.cat([x32, x32[:,:,-1].unsqueeze(-1).repeat(1,1,self.output_n)], dim=2)
            
            x22 = x32[:, self.dim_used, :]
            x12 = self.down(x22, self.Index2212)
            x7 = self.down(x12, self.Index127)
            x4 = self.down(x7, self.Index74)

            x32 = dct_transform_torch(x32, self.dct_m, self.dct_used)
            x22 = dct_transform_torch(x22, self.dct_m, self.dct_used)
            x12 = dct_transform_torch(x12, self.dct_m, self.dct_used)
            x7 = dct_transform_torch(x7, self.dct_m, self.dct_used)
            x4 = dct_transform_torch(x4, self.dct_m, self.dct_used)

            x32 = (x32-self.global_min)/(self.global_max-self.global_min)
            x22 = (x22-self.global_min)/(self.global_max-self.global_min)
            x12 = (x12-self.global_min)/(self.global_max-self.global_min)
            x7 = (x7-self.global_min)/(self.global_max-self.global_min)
            x4 = (x4-self.global_min)/(self.global_max-self.global_min)

            x32=x32*2-1
            x22=x22*2-1
            x12=x12*2-1
            x7=x7*2-1
            x4=x4*2-1

            # print(x32.shape, x22.shape, x12.shape, x7.shape, x4.shape)

            # extend inputs + dct + global min and max
            return {
                "p32":x32,
                "p22":x22,
                "p12":x12,
                "p7":x7,
                "p4":x4
            }
        else:
            return x
