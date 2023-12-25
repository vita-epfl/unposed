import logging
import os
import re
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
from utils.others import DPWconvertTo3D

from path_definition import PREPROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class Preprocessor3DPW:
    def __init__(self, dataset_path,
                 custom_name, load_60Hz=False):

        self.dataset_path = dataset_path
        self.custom_name = custom_name
        self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, '3DPW_total')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.load_60Hz = load_60Hz

    def normal(self, data_type='train'):
        logger.info('start creating 3DPW normal static data ... ')
        
        const_joints = np.arange(4 * 3)
        var_joints = np.arange(4 * 3, 22 * 3)

        if self.custom_name:
            output_file_name = f'{data_type}_xyz_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_xyz_3dpw.jsonl'

        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"

  
        self.dataset_path = os.path.join(self.dataset_path, data_type)

        for entry in os.scandir(self.dataset_path):
            if not entry.name.endswith('.pkl'):
                continue
            logger.info(f'file name: {entry.name}')
            pickle_obj = pd.read_pickle(entry.path)
            video_name = re.search('(\w+).pkl', entry.name).group(1)
            if self.load_60Hz:
                pose_data = np.array(pickle_obj['poses_60Hz'])
            else:
                pose_data = np.array(pickle_obj['jointPositions'])

            pose_data = DPWconvertTo3D(pose_data) * 1000

            total_frame_num = pose_data.shape[1]

            data = []
            video_data = {
                'obs_pose': defaultdict(list)
            }
            for j in range(total_frame_num):
                for p_id in range(pose_data.shape[0]):
                    video_data['obs_pose'][p_id].append(
                        pose_data[p_id, j, :].tolist()
                    )
            if len(list(video_data['obs_pose'].values())) > 0:
                for p_id in range(len(pose_data)):
                    data.append([
                        '%s-%d' % (video_name, 0),
                        video_data['obs_pose'][p_id]
                    ] if not self.load_60Hz else [
                        '%s-%d' % (video_name, 0),
                        np.array(video_data['obs_pose'][p_id])[:, var_joints].tolist(),
                        np.array(video_data['obs_pose'][p_id])[:, const_joints].tolist()
                    ])
            with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                for data_row in data:
                    if not self.load_60Hz:
                        writer.write({
                            'video_section': data_row[0],
                            'xyz_pose': data_row[1],
                        })
                    else:
                        writer.write({
                            'video_section': data_row[0],
                            'xyz_pose': data_row[1],
                            'xyz_const_pose': data_row[2],
                        })
