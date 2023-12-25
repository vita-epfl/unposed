from .functions import *
from .dataset_utils import JOINTS_TO_INCLUDE, SCALE_RATIO

LOSS_K, UNC_K = 'loss', 'uncertainty'


def entropy_uncertainty(p, as_list=False):
    entropy = - p * torch.log2(p)
    out = torch.nansum(entropy, dim=1)
    if as_list:
        return out
    return torch.mean(out)


def calculate_pose_uncertainty(prediction_pose, dc_model, dataset_name: str):
    uncertainties = 0
    with torch.no_grad():
        p, _ = dc_model(scale(prediction_pose[:, :, JOINTS_TO_INCLUDE[dataset_name]], SCALE_RATIO[dataset_name]))
        uncertainties += entropy_uncertainty(p)
    return uncertainties
