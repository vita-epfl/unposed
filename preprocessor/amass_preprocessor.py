import logging
import os

import jsonlines
import numpy as np
from utils.others import AMASSconvertTo3D

from path_definition import PREPROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

amass_splits = {
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
    'validation': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['BioMotionLab_NTroje'],
}

class AmassPreprocessor:
    def __init__(self, dataset_path,
                 custom_name):

        self.dataset_path = dataset_path
        self.custom_name = custom_name
        self.output_dir = os.path.join(PREPROCESSED_DATA_DIR, 'AMASS_total')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def normal(self, data_type='train'):
        logger.info('start creating AMASS normal static data ... ')
        const_joints = np.arange(4 * 3)
        var_joints = np.arange(4 * 3, 22 * 3)

        if self.custom_name:
            output_file_name = f'{data_type}_xyz_{self.custom_name}.jsonl'
        else:
            output_file_name = f'{data_type}_xyz_AMASS.jsonl'
        
        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"

        assert data_type in amass_splits, "data type must be one of train, validation or test"
        
        dataset_names = amass_splits[data_type]
        for dataset_name in dataset_names:
            raw_dataset_name = dataset_name
            logger.info(f'dataset name: {dataset_name}')
            for sub in os.listdir(os.path.join(self.dataset_path, dataset_name)):
                raw_sub = sub
                logger.info(f'subject name: {sub}')
                sub = os.path.join(self.dataset_path, dataset_name, sub)
                if not os.path.isdir(sub):
                    continue
                for act in os.listdir(sub):
                    if not act.endswith('.npz'):
                        continue
                    raw_act = act[:-4]
                    pose_all = np.load(os.path.join(sub, act))
                    try:
                        pose_data = pose_all['poses']
                    except:
                        print('no poses at {} {}'.format(sub, act))
                        continue

                    pose_data = AMASSconvertTo3D(pose_data) # shape = [num frames , 66]
                    pose_data = pose_data * 1000 # convert from m to mm

                    total_frame_num = pose_data.shape[0]

                    data = []

                    video_data = {
                        'obs_pose': list(),
                        'obs_const_pose': list(),
                        'fps': int(pose_all['mocap_framerate'].item())
                    }

                    for j in range(total_frame_num):
                        video_data['obs_pose'].append(pose_data[j][var_joints].tolist())
                        video_data['obs_const_pose'].append(pose_data[j][const_joints].tolist())

                    data.append([
                        '%s-%d' % ("{}-{}-{}".format(raw_dataset_name, raw_sub, raw_act), 0),
                        video_data['obs_pose'], video_data['obs_const_pose'],
                        video_data['fps']
                    ])
                        
                with jsonlines.open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                    for data_row in data:
                        writer.write({
                            'video_section': data_row[0],
                            'xyz_pose': data_row[1],
                            'xyz_const_pose': data_row[2],
                            'fps': data_row[3]
                        })
