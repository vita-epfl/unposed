import os
import numpy as np
import spacepy.pycdf
import json

from path_definition import PREPROCESSED_DATA_DIR
import logging
logger = logging.getLogger(__name__)

SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8', 'S9'],
    'validation': ['S11'],
    'test': ['S5']
}


class Human36m2dPreprocessor:
    def __init__(self, dataset_path,
                 custom_name):
        self.dataset_path = dataset_path
        self.custom_name = custom_name
        self.output_dir = os.path.join(
            PREPROCESSED_DATA_DIR, 'human36m2d'
        )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def normal(self, data_type='train'):
        self.subjects = SPLIT[data_type]
        logger.info(
            'start creating Human3.6m2d preprocessed data from Human3.6m2d dataset ... ')

        if self.custom_name:
            output_file_name = \
                f'{data_type}_{self.custom_name}.jsonl'
        else:
            output_file_name = \
                f'{data_type}_human3.6m.jsonl'

        assert os.path.exists(os.path.join(
            self.output_dir,
            output_file_name
        )) is False, f"preprocessed file exists at {os.path.join(self.output_dir, output_file_name)}"

        for subject in self.subjects:
            logger.info("handling subject: {}".format(subject))
            folder_path = f"{subject}/MyPoseFeatures/D2_Positions/"
            full_path = os.path.join(self.dataset_path, folder_path)

            # Ensure the directory exists
            if not os.path.exists(full_path):
                continue
            
            # Open or create the jsonl file and write data
            with open(os.path.join(self.output_dir, output_file_name), 'a') as writer:
                # Process each file in the directory
                for file in os.listdir(full_path):
                    if file.endswith('.cdf'):
                        data_dict = self.process_cdf_files(full_path, file)
                        writer.write(json.dumps(data_dict) + '\n')

    def process_cdf_files(self, path, file):
        cdf = spacepy.pycdf.CDF(path + file)
        pose_data = cdf['Pose'][:]
        cdf.close()
        
        # Extract the action from filename
        action = file.split('.')[0].split(' ')[0].lower()

        pose_data = pose_data[0]

        # # go from 32 -> 13 joints for use with openpifpaf
        # ignored_joints = [0, 4, 5, 9, 10, 11, 12, 13, 15, 18, 20, 21, 22, 23, 26, 28, 29, 30, 31]
        # ignored_joints = [2 * i for i in ignored_joints]
        # mask = np.ones(64, dtype=bool)
        # mask[ignored_joints] = False
        # mask[list(np.array(ignored_joints) + 1)] = False
        
        # Create dictionary for json line
        data_dict = {
            "video_section": file,
            "action": action,
            # "xyz_pose": np.array(pose_data)[:, mask].tolist()
            "xyz_pose": np.array(pose_data)
            .tolist()
        }
        
        return data_dict