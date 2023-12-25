import os
import pickle as pkl
from os import walk

import numpy as np
import torch
from torch.utils.data import Dataset

from . import ang2joint
from ..utils.dataset_utils import JOINTS_TO_INCLUDE, SKIP_RATE
from ..utils.functions import scale

DPW3_DIM_USED = JOINTS_TO_INCLUDE['AMASS']
DPW3_SKIP_RATE = SKIP_RATE['AMASS']
'''
adapted from 
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/dpw3d.py
'''

PATH_TO_SMPL_SKELETON = '../utils/smpl_skeleton.npz'
SCALE_RATIO = 1


class Dpw3(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, apply_joints_to_include=False, split=0):
        """

        Args:
            data_dir:
            input_n:
            output_n:
            skip_rate:
            apply_joints_to_include:
            split:
        """
        self.path_to_data = os.path.join(data_dir, '3DPW/sequenceFiles')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.apply_joints_to_include = apply_joints_to_include
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        seq_len = self.in_n + self.out_n

        if split == 0:
            data_path = self.path_to_data + '/train/'
        elif split == 2:
            data_path = self.path_to_data + '/test/'
        elif split == 1:
            data_path = self.path_to_data + '/validation/'
        files = []
        for (dirpath, dirnames, filenames) in walk(data_path):
            files.extend(filenames)
        skel = np.load(PATH_TO_SMPL_SKELETON)
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()[:, :22]
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            parent[i] = parents[i]
        n = 0

        sample_rate = int(60 // 25)

        for f in files:
            with open(data_path + f, 'rb') as f:
                print('>>> loading {}'.format(f))
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['poses_60Hz']
                for i in range(len(joint_pos)):
                    poses = joint_pos[i]
                    fn = poses.shape[0]
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    poses = poses[:, :-2]
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                    self.p3d.append(p3d.cpu().data.numpy())

                    if split == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        pose = self.p3d[key][fs].reshape((-1, 66))[:, DPW3_DIM_USED] if self.apply_joints_to_include \
            else self.p3d[key][fs].reshape((-1, 66))
        return scale(pose, SCALE_RATIO), item  # , key


