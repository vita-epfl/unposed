import os
from argparse import Namespace

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from .model.dc.deep_clustering import DCModel
from .model.dc.train_dc_seq import train_dc_model, cluster
from .model.lstm.lstm import LstmAutoEncoder, EncoderWrapper
from .model.lstm.train_lstm import train_lstm_model
from .utils.train_utils import save_model, save_model_results_dict
from .utils.uncertainty import *
from .utils.dataset_utils import TRAIN_K, VALID_K, TEST_K, INCLUDED_JOINTS_COUNT, SKIP_RATE, SCALE_RATIO, H36_ACTIONS, \
    DIM


def load_dc_model(dataset_name: str, n_clusters: int, dc_model_path: str, dev='cuda'):
    lstm_ae = LstmAutoEncoder(pose_dim=INCLUDED_JOINTS_COUNT[dataset_name]).to(dev)
    dc_model = DCModel(lstm_ae, n_clusters=n_clusters).to(dev)
    dc_model.load_state_dict(torch.load(dc_model_path))
    return dc_model