import logging
import os
import pickle
from itertools import chain

import torch

from models import MODELS
from losses import LOSSES
from optimizers import OPTIMIZERS

logger = logging.getLogger(__name__)


def load_snapshot(snapshot_path):
    snapshot = torch.load(snapshot_path, map_location='cpu')
    model = MODELS[snapshot['model_args'].type](snapshot['model_args'])
    model.load_state_dict(snapshot['model_state_dict'])
    loss_module = LOSSES[snapshot['loss_args'].type](snapshot['loss_args']) 
    loss_module.load_state_dict(snapshot['loss_state_dict'])
    optimizer = OPTIMIZERS[snapshot['optimizer_args'].type](chain(model.parameters(), loss_module.parameters()), snapshot['optimizer_args'])
    optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    return (model, loss_module, optimizer, snapshot['optimizer_args'], snapshot['epoch'], snapshot['train_reporter'],
            snapshot['valid_reporter'])


def save_snapshot(model, loss_module, optimizer, optimizer_args, epoch, train_reporter, valid_reporter, save_path, best_model=False):
    logger.info('### Taking Snapshot ###')
    snapshot = {
        'model_state_dict': model.state_dict(),
        'model_args': model.args,
        'loss_state_dict': loss_module.state_dict(),
        'loss_args': loss_module.args,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_args': optimizer_args,
        'epoch': epoch,
        'train_reporter': train_reporter,
        'valid_reporter': valid_reporter
    }
    if not best_model:
        torch.save(snapshot, os.path.join(save_path, 'snapshots', '%d.pt' % epoch))
    else:
        torch.save(snapshot, os.path.join(save_path, 'snapshots', 'best_model.pt'))
    del snapshot


def save_test_results(result_df, result_tensor, save_dir):
    result_df.to_csv(os.path.join(save_dir, 'generated_outputs', 'results.csv'), index=False)
    with open(os.path.join(save_dir, 'generated_outputs', 'results.pkl'), 'wb') as f:
        pickle.dump(result_tensor, f)


def setup_training_dir(parent_dir):
    os.makedirs(os.path.join(parent_dir, 'snapshots'), exist_ok=False)
    os.makedirs(os.path.join(parent_dir, 'plots'), exist_ok=False)
    os.makedirs(os.path.join(parent_dir, 'metrics_history'), exist_ok=False)


def setup_testing_dir(parent_dir):
    os.makedirs(os.path.join(parent_dir, 'generated_outputs'), exist_ok=False)


def setup_visualization_dir(parent_dir):
    vis_dir = os.path.join(parent_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir
