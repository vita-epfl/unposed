import torch
from torch.utils.data import Dataset
import random
import os

import utils.functions as data_utils
import numpy as np
from math import *
from utils.dataset_utils import *


class Human36M(Dataset):

    def __init__(self, path_to_data: str, input_n: int, output_n: int, skip_rate: int, scale: float, actions: list,
                 apply_joints_to_include=False, vel=False,
                 split=0):
        """
        :param path_to_data:
        :param actions: a list of ACTIONS. All ACTIONS will be considered if None
        :param input_n: number of input frames
        :param output_n: number of output frames
        :param split: 0 train, 1 testing, 2 validation
        :param skip_rate:
        """
        self.path_to_data = os.path.join(path_to_data, 'h3.6m/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.apply_joints_to_include = apply_joints_to_include
        self.skip_rate = 2
        self.p3d = {}
        self.data_idx = []
        self.scale = scale
        seq_len = self.in_n + self.out_n
        self.vel = vel if split != 2 else False
        subs = H36M_SUBJECTS[split]
        key = 0
        for subj in subs:
            for index in np.arange(len(actions)):
                action = actions[index]
                if self.split == 0 or self.split == 1:
                    for sub_action in [1, 2]:  # subactions
                        self._init_train_or_val_set(action, key, seq_len, skip_rate, sub_action, subj)
                        key += 1
                else:
                    self._init_test(action, key, seq_len, subj)
                    key += 2
        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
        self.len = self.__len__()

    def _init_test(self, action, key, seq_len, subj):
        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
        filename = self._get_file_name(action, 1, subj)
        num_frames1, the_sequence1 = self._get_sequence(filename)
        coordinates = self._get_3d_coordinates(num_frames1, the_sequence1)
        self.p3d[key] = coordinates, action
        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
        filename = self._get_file_name(action, 2, subj)
        num_frames2, the_sequence2 = self._get_sequence(filename)
        coordinates = self._get_3d_coordinates(num_frames2, the_sequence2)
        self.p3d[key + 1] = coordinates, action
        fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                                                       input_n=self.in_n)
        valid_frames = fs_sel1[:, 0]
        tmp_data_idx_1 = [key] * len(valid_frames)
        tmp_data_idx_2 = list(valid_frames)
        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
        valid_frames = fs_sel2[:, 0]
        tmp_data_idx_1 = [key + 1] * len(valid_frames)
        tmp_data_idx_2 = list(valid_frames)
        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

    def _init_train_or_val_set(self, action, key, seq_len, skip_rate, sub_action, subj):
        print(f'Reading subject {subj}, action {action}, subaction {sub_action}')
        filename = self._get_file_name(action, sub_action, subj)
        num_frames, the_sequence = self._get_sequence(filename)
        coordinates = self._get_3d_coordinates(num_frames, the_sequence)
        if self.vel:
            coordinates = coordinates[1:] - coordinates[:-1]
            num_frames -= 1
            seq_len -= 1
        self.p3d[key] = coordinates, action
        valid_frames = np.arange(0, num_frames - seq_len + 1, skip_rate)
        tmp_data_idx_1 = [key] * len(valid_frames)
        tmp_data_idx_2 = list(valid_frames)
        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

    def _get_3d_coordinates(self, num_frames, the_sequence):
        self._remove_extra_params(the_sequence)
        # convert exponential map format to 3D points
        p3d = data_utils.expmap2xyz_torch(the_sequence)  # shape: (frames, 32, 3)
        return p3d.view(num_frames, -1).cpu().data.numpy()  # shape: (frames, 96)

    def _remove_extra_params(self, the_sequence):
        # remove global rotation and translation
        the_sequence[:, 0:6] = 0

    def _get_sequence(self, filename):
        the_sequence = data_utils.readCSVasFloat(filename)
        n, d = the_sequence.shape  # n = number of frames, d = number of parameters
        frames_with_skip = range(0, n, self.skip_rate)
        num_frames = len(frames_with_skip)
        the_sequence = np.array(the_sequence[frames_with_skip, :])
        the_sequence = torch.from_numpy(the_sequence).float()
        if torch.cuda.is_available():
            the_sequence = the_sequence.cuda()
        return num_frames, the_sequence

    def _get_file_name(self, action, sub_action, subj):
        return '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, sub_action)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        n_key, n_start_frame = self.data_idx[min(item + 1, self.len - 1)]
        seq_len = self.in_n + self.out_n
        if self.vel:
            seq_len -= 1
        fs = np.arange(start_frame, start_frame + seq_len)
        n_fs = np.arange(n_start_frame, n_start_frame + seq_len)
        pose, action = self.p3d[key]
        n_pose, n_action = self.p3d[n_key]
        pose, n_pose = pose[fs], n_pose[n_fs]
        if self.apply_joints_to_include:
            pose = pose[..., np.array(JOINTS_TO_INCLUDE['Human36m'])]
            n_pose = n_pose[..., np.array(JOINTS_TO_INCLUDE['Human36m'])]
        return data_utils.scale(pose, self.scale), action, data_utils.scale(n_pose, self.scale), n_action, item