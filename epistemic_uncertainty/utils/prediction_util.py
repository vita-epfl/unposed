import torch
from torch.utils.data import DataLoader

from .dataset_utils import DIM, JOINTS_TO_INCLUDE, INCLUDED_JOINTS_COUNT
from .functions import rescale_to_original_joints_count
from ..model.sts_gcn.sts_gcn import STSGCN
from ..model.pgbig.stage_4 import MultiStageModel
from ..model.zerovel.zerovel import Zerovel

IDX, INP_K, OUT_K, GT_K = 'index', 'inputs', 'outputs', 'ground_truths'
RJ, NRJ, ABS, DIFF, RT, CMP, ABS_ROC, DIFF_ROC, CLS_MAP = 'Rejected', 'Non-rejected', 'Uncertainty', 'Self-uncertainty', 'Rate', 'Comparing', 'Uncertainty AUROC', 'Self-uncertainty AUROC', 'Clusters transition map'
PRED_MODELS = {'zerovel': Zerovel, 'sts': STSGCN, 'pgbig': MultiStageModel}
PRED_MODELS_ARGS = {
    'sts': {'input_channels': DIM, 'input_time_frame': 10, 'output_time_frame': 25, 'st_gcnn_dropout': 0.1,
            'joints_to_consider': 22, 'n_txcnn_layers': 4, 'txc_kernel_size': [DIM, DIM], 'txc_dropout': 0.0},
    'pgbig': {'in_features': 66, 'num_stages': 12, 'd_model': 16, 'kernel_size': 10, 'drop_out': 0.3, 'input_n': 10, 'output_n': 25, 'dct_n': 35, 'cuda_idx': 0}
    }


def get_prediction_model_dict(model, data_loader: DataLoader, input_n: int, output_n: int, dataset_name: str, vel=False,
                              dev='cuda', dropout=False) -> dict:
    prediction_dict = {INP_K: [], OUT_K: [], GT_K: []}
    if dropout:
        enable_dropout(model)
    for _, data_arr in enumerate(data_loader):
        pose = data_arr[0].to(dev)
        B = pose.shape[0]
        inp_seq = pose[:, :output_n, :]
        if len(inp_seq) == 1:
            inp_seq = inp_seq.unsqueeze(0)
        inp = pose[:, output_n - input_n:output_n, JOINTS_TO_INCLUDE[dataset_name]]. \
            view(B, input_n, INCLUDED_JOINTS_COUNT[dataset_name] // DIM, DIM).permute(0, 3, 1, 2)
        gt = pose[:, output_n:, :]
        print(f'GT: {gt.shape}')
        if len(gt) == 1:
            gt = gt.unsqueeze(0)
        with torch.no_grad():
            out = model(inp).permute(0, 1, 3, 2).contiguous().view(-1, output_n, INCLUDED_JOINTS_COUNT[dataset_name])
            out = rescale_to_original_joints_count(out, gt, dataset_name)
            print(out.shape)
#            prediction_dict['outputs_orig'].append(out)
            if vel:
                inp_seq = inp_seq[:, 1:, :] - inp_seq[:, :-1, :]
                out = out[:, 1:, :] - out[:, :-1, :]
#                gt = gt[:, 1:, :] - gt[:, :-1, :]
            prediction_dict[GT_K].append(gt)
            prediction_dict[OUT_K].append(out)
            prediction_dict[INP_K].append(inp_seq)
    prediction_dict[GT_K] = torch.concat(prediction_dict[GT_K], dim=0)
    prediction_dict[OUT_K] = torch.concat(prediction_dict[OUT_K], dim=0)
    prediction_dict[INP_K] = torch.concat(prediction_dict[INP_K], dim=0)
#    prediction_dict['outputs_orig'] = torch.concat(prediction_dict['outputs_orig'], dim=0)
    return prediction_dict

def get_dataloader_dict(data_loader: DataLoader, input_n: int, output_n: int, dataset_name: str, vel=False, dev='cuda') -> dict:
    prediction_dict = {GT_K: [], INP_K: []}
    for _, data_arr in enumerate(data_loader):
        pose = data_arr[0].to(dev)
        inp_seq = pose[:, :output_n, :]
        if len(inp_seq) == 1:
            inp_seq = inp_seq.unsqueeze(0)
        B = pose.shape[0]
        if pose.shape[1] > output_n:
            gt = pose[:, output_n:, :]
        else:
            gt = pose
        if len(gt) == 1:
            gt = gt.unsqueeze(0)
        if vel:
            inp_seq = inp_seq[:, 1:, :] - inp_seq[:, :-1, :]
            gt = gt[:, 1:, :] - gt[:, :-1, :]
        prediction_dict[GT_K].append(gt)
        prediction_dict[INP_K].append(inp_seq)
    prediction_dict[GT_K] = torch.concat(prediction_dict[GT_K], dim=0)
    prediction_dict[INP_K] = torch.concat(prediction_dict[INP_K], dim=0)
    return prediction_dict

def enable_dropout(m):
    for each_module in m.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            print('Dropout enabled.')
            each_module.train()