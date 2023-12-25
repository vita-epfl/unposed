from torch.utils.data import DataLoader

from .pose_dataset import PoseDataset

DATASETS = ['3dpw', 'human3.6m', 'amass']
DATA_TYPES = ['train', 'validation', 'test']
VISUALIZING_TYPES = ['observed', 'future', 'predicted', 'completed']


def get_dataloader(dataset_path, args):
    if dataset_path is None:
        return None
    
    dataset = PoseDataset(
        dataset_path, args.keypoint_dim, args.is_testing, args.model_pose_format, args.metric_pose_format, 
        args.seq_rate, args.frame_rate, args.len_observed, 
        args.len_future, args.is_h36_testing, args.random_reverse_prob
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=args.pin_memory,
                            num_workers=args.num_workers)
    return dataloader

def get_dataset(dataset_path, args):
    if dataset_path is None:
        return None

    dataset = PoseDataset(
        dataset_path, args.keypoint_dim, args.is_testing, args.model_pose_format, args.metric_pose_format,
        args.seq_rate, args.frame_rate, args.len_observed,
        args.len_future, args.is_h36_testing, args.random_reverse_prob
    )
    return dataset
