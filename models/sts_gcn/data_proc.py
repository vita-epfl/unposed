import numpy as np
import torch
from torch import nn


def joint_to_index(x):
    return np.concatenate((x * 3, x * 3 + 1, x * 3 + 2))


class AMASS_3DPW_values():
    def __init__(self) -> None:
        self.mean = np.array([[[ 144.42061,   -471.00433,     42.905945,  -144.2189,    -471.00433,
        55.37049,      5.0257893,  235.50217,      6.188506,   131.57523,
        -916.6374,     -73.56259,   -129.137,     -914.8206,     -67.33688,
            5.8155527,  278.29196,     43.45168,    189.25381,   -942.00867,
        31.58287,   -185.46811,   -942.00867,     38.146023,     1.6395822,
        504.07196,     65.04476,     90.96787,    467.06006,     46.06653,
        -79.57573,    464.56763,     35.583405,     5.7321978,  544.3716,
        132.00195,    189.30196,    464.18073,     46.495617,  -181.78586,
        461.8248,      38.285446,   242.19247,    208.86894,     10.837954,
        -243.21066,    220.56078,     20.73184,    256.45264,     66.64482,
        116.55112,   -262.37643,     88.037315,   129.05185  ]]])
        self.std = np.array([[[ 48.352272,  68.6015,   119.17078,   47.49278,   69.03037,  120.7153,
        9.933628,  16.11266,   32.15347,   81.19508,  148.52235,  160.55476,
        78.806435, 148.95927,  161.60782,   15.046006,  26.999517,  41.232426,
        90.12439,  126.81438,  174.0965,    87.97808,  128.31987,  173.8965,
        38.010742,  43.834454,  91.36834,   28.467258,  66.382034,  72.003075,
        26.970959,  66.33471,   69.758385,  48.895977,  62.34938,  100.590385,
        33.747925,  76.94056,   85.15882,   32.314583,  77.06175,   83.645386,
        80.88272,  109.25045,  123.5628,    79.029915, 115.18032,  127.966995,
        158.545,    217.86617,  164.67949,  156.79645,  235.94897,  175.8384  ]]])

class Human36m_values():
    def __init__(self) -> None:
        self.dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

        self.mean = np.array([[[-107.9520, -334.9428, 159.4178, -59.2900, -708.5010, 61.4318,
                        -61.6391, -757.3103, 189.2157, -72.0266, -761.0327, 251.2684,
                        151.7538, -326.8112, 161.4840, 134.0947, -709.5225, 71.7927,
                        153.4157, -744.0466, 200.7195, 163.8421, -737.7441, 261.8018,
                        -17.9565, 210.8857, -12.5731, -30.5735, 429.6271, 36.1767,
                        -43.2606, 489.0777, 114.7583, -54.7775, 578.9327, 88.4990,
                        108.9527, 394.7472, 26.0654, 237.0195, 213.8401, 44.9180,
                        188.2216, 135.0727, 139.9878, 152.3083, 163.3067, 155.1163,
                        196.3242, 118.3158, 182.5405, -163.6815, 375.3079, 23.2578,
                        -266.1268, 186.6490, 53.2938, -217.2098, 156.2352, 160.8916,
                        -200.4095, 191.2718, 165.2301, -223.5151, 149.2325, 211.9896]]])
        self.std = np.array([[[65.4117, 166.9468, 160.5147, 109.2458, 295.7622, 210.9699, 122.5746,
                        308.4443, 228.9709, 131.0754, 310.0372, 235.9644, 74.9162, 174.3366,
                        163.9575, 129.1666, 296.0691, 209.0041, 154.0681, 305.1154, 224.3635,
                        165.3411, 304.1239, 230.2749, 19.6905, 71.2422, 64.0733, 52.6362,
                        150.2302, 141.1058, 68.3720, 177.7844, 164.2342, 78.0215, 203.7356,
                        192.8816, 47.0527, 137.0687, 138.8337, 72.1145, 127.8964, 170.1875,
                        151.9798, 210.0934, 199.3142, 155.3852, 219.3135, 193.1652, 191.3546,
                        254.2903, 225.2465, 45.0912, 135.5994, 133.7429, 74.3784, 133.9870,
                        160.7077, 143.9800, 235.9862, 196.2391, 147.1276, 232.4836, 188.2000,
                        189.1858, 308.0274, 235.1181]]])

        index_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        self.index_to_ignore = joint_to_index(index_to_ignore)

        index_to_equal = np.array([13, 19, 22, 13, 27, 30])
        self.index_to_equal = joint_to_index(index_to_equal)

        index_to_copy = np.array([0, 1, 6, 11])
        self.index_to_copy = joint_to_index(index_to_copy)

human36m = Human36m_values()
amass_3dpw = AMASS_3DPW_values()


class Human36m_Preprocess(nn.Module):
    def __init__(self, args):
        super(Human36m_Preprocess, self).__init__()
        self.args = args
        self.mean = torch.tensor(human36m.mean).to(args.device).float()
        self.std = torch.tensor(human36m.std).to(args.device).float()

    def forward(self, observed_pose, normal=True):
        observed_pose = observed_pose[:, :, human36m.dim_used]
        if normal:
            observed_pose = (observed_pose - self.mean) / self.std
        return observed_pose


class Human36m_Postprocess(nn.Module):
    def __init__(self, args):
        super(Human36m_Postprocess, self).__init__()
        self.args = args
        self.mean = torch.tensor(human36m.mean).to(args.device).float()
        self.std = torch.tensor(human36m.std).to(args.device).float()

    def forward(self, observed_pose, pred_pose, normal=True):
        if normal:
            pred_pose = (pred_pose * self.std) + self.mean

        x = torch.zeros([pred_pose.shape[0], pred_pose.shape[1], 96]).to(self.args.device)
        x[:, :, human36m.dim_used] = pred_pose
        x[:, :, human36m.index_to_copy] = observed_pose[:, -1:, human36m.index_to_copy]
        x[:, :, human36m.index_to_ignore] = x[:, :, human36m.index_to_equal]
        return x

class AMASS_3DPW_Preprocess(nn.Module):
    def __init__(self, args):
        super(AMASS_3DPW_Preprocess, self).__init__()
        self.args = args
        self.mean = torch.tensor(amass_3dpw.mean).to(args.device).float()
        self.std = torch.tensor(amass_3dpw.std).to(args.device).float()

    def forward(self, observed_pose, normal=True):
        if normal:
            observed_pose = (observed_pose - self.mean) / self.std
        return observed_pose


class AMASS_3DPW_Postprocess(nn.Module):
    def __init__(self, args):
        super(AMASS_3DPW_Postprocess, self).__init__()
        self.args = args
        self.mean = torch.tensor(amass_3dpw.mean).to(args.device).float()
        self.std = torch.tensor(amass_3dpw.std).to(args.device).float()

    def forward(self, observed_pose, pred_pose, normal=True):
        if normal:
            pred_pose = (pred_pose * self.std) + self.mean
        return pred_pose

class Preprocess(nn.Module):
    def __init__(self, args):
        super(Preprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose, normal=True):
        return observed_pose

class Postprocess(nn.Module):
    def __init__(self, args):
        super(Postprocess, self).__init__()
        self.args = args

    def forward(self, observed_pose, pred_pose, normal=True):
        return pred_pose