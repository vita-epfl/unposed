import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

from .ang2joint import *
from ..utils.functions import scale
from ..utils.dataset_utils import JOINTS_TO_INCLUDE, SKIP_RATE

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/amass3d.py
'''

PATH_TO_SMPL_SKELETON = '../utils/smpl_skeleton.npz'
AMASS_DIM_USED = JOINTS_TO_INCLUDE['AMASS']
AMASS_SKIP_RATE = SKIP_RATE['AMASS']
AMASS_SCALE_RATIO = 1


class Amass(Dataset):

    def __init__(self, path_to_data, input_n, output_n, skip_rate, split=0,
                 apply_joints_to_include=False, ):
        """

        Args:
            path_to_data:
            input_n:
            output_n:
            skip_rate:
            apply_dim_used:
            split:
        """
        self.path_to_data = os.path.join(path_to_data, 'AMASS/')  # "D:\data\AMASS\\"
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.skip_rate = skip_rate
        self.apply_joints_to_include = apply_joints_to_include

        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)  # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n
        #TODO: DELETE SFU FROM TEST AND VAL
        amass_splits = [
            ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
            ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
            ['BioMotionLab_NTroje'],
        ]

        # load mean skeleton
        skel = np.load(PATH_TO_SMPL_SKELETON)
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()[:, :22]
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            parent[i] = parents[i]
        n = 0
        for ds in amass_splits[split]:
            if not os.path.isdir(self.path_to_data + ds):
                print(ds)
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + '/' + ds):
                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    if not act.endswith('.npz'):
                        continue
                    # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                    #     continue
                    pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + act)
                    try:
                        poses = pose_all['poses']
                    except:
                        print('no poses at {}_{}_{}'.format(ds, sub, act))
                        continue
                    frame_rate = pose_all['mocap_framerate']
                    fn = poses.shape[0]
                    sample_rate = int(frame_rate // 25)
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint(p3d0_tmp, poses, parent)
                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    self.p3d.append(p3d.cpu().data.numpy())
                    if split == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, self.skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    self.keys.append((ds, sub, act))
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        pose = self.p3d[key][fs].reshape((-1, 66))[:, AMASS_DIM_USED] \
            if self.apply_joints_to_include \
            else self.p3d[key][fs].reshape((-1, 66))
        return scale(pose, AMASS_SCALE_RATIO), item  # , key


if __name__ == '__main__':
    a = Amass('../dataset/', 0, 25, AMASS_SKIP_RATE, split=0, apply_joints_to_include=True)
    d = DataLoader(a, batch_size=256, shuffle=True,
               pin_memory=True)
    for data in d:
        pass
