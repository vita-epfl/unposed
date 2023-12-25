import copy
import math
import time
from argparse import Namespace

import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from .deep_clustering import DCModel
from ..lstm.lstm import EncoderWrapper
from ...utils.train_utils import *


def train_dc_model(args: Namespace, dc_model: DCModel, dataset, batch_size, num_workers=8, optimizer=None,
                   scheduler=None, dev='cuda'):
    """
    By now, the following should have been completed:
    Step 1: Pretraining the AE
    Step 2: Initializing Clusters with K-Means
    Now:
    Step 3: Deep Clustering
    """
    params = list(dc_model.parameters()) + list(dc_model.clustering_layer.parameters())
    if optimizer is None:
        new_lr, optimizer = get_optimizer(args, optimizer, params)
    else:
        new_lr, optimizer = args.dc_lr, optimizer(params)

    loader_args = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True, }
    dataset.return_indices = True
    e_loader = DataLoader(dataset, shuffle=False, **loader_args)
    m_loader = DataLoader(dataset, shuffle=True, drop_last=True, **loader_args)

    reconstruction_loss_fn = nn.MSELoss(reduction='mean').to(dev)
    cls_loss_fn = nn.KLDivLoss().to(dev)
    seq_loss_fn = nn.MSELoss(reduction='mean').to(dev)

    y_pred = []
    y_pred_prev = np.copy(y_pred)
    loss = [0, 0, 0]
    p = None

    dc_model.train()
    dc_model = dc_model.to(dev)

    optimizer.zero_grad()
    stop_flag = False

    eval_epochs, eval_iterations = get_eval_epochs_iterations(args, m_loader)
    b_loss = 10000
    b_model = None

    for epoch in range(args.dc_epochs):
        loss_list = []
        start_time = time.time()
        for it, data_arr in enumerate(tqdm(m_loader, desc="Train Epoch", leave=True)):
            if (epoch % eval_epochs == 0) or (epoch == 0):
                if it % eval_iterations == 0:
                    print("Target distribution update at epoch {} iteration {}".format(epoch, it))
                    p, y_pred = calc_curr_p(dc_model, e_loader)

                    if epoch >= 1:
                        stop_flag, y_pred_prev, delta_label = eval_clustering_stop_cret(y_pred, y_pred_prev,
                                                                                        stop_cret=args.dc_stop_cret)
                        if epoch >= 3 and stop_flag:
                            print("Stop flag in epoch {}".format(epoch))
                            break
                        else:
                            stop_flag = False

            if not stop_flag:
                indices = data_arr[-1]
                p_iter = torch.from_numpy(p[indices]).to(dev)
                data = data_arr[0].to(dev)
                n_data = data_arr[2].to(dev)
                cls_softmax, x_reconstructed, z = dc_model(data, ret_z=True)
                n_cls, _, n_z = dc_model(n_data, ret_z=True)
                reconstruction_loss = reconstruction_loss_fn(data, x_reconstructed)
                clustering_loss = cls_loss_fn(torch.log(p_iter), cls_softmax)
                actions = np.array(data_arr[1])
                n_actions = np.array(data_arr[3])
                idx = np.where(n_actions == actions)[0]
                seq_loss = seq_loss_fn(n_z[idx], z[idx])

                reg_loss = calc_reg_loss(dc_model)
                loss = reconstruction_loss + args.dc_gamma * clustering_loss + args.alpha * reg_loss + args.dc_lambda * seq_loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                c_loss = loss.item()
                if b_loss > c_loss:
                    c_loss = b_loss
                    b_model = copy.deepcopy(dc_model)
                loss_list.append(c_loss)

        new_lr = adjust_lr(optimizer, epoch, new_lr, args.dc_lr_decay, scheduler=scheduler)
        print("Epoch {} Done in {}s, loss is {}\n".format(epoch, time.time() - start_time, loss))
        if stop_flag:
            break
    return b_model


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def get_optimizer(args, opt, params):
    if opt is None:
        opt = torch.optim.Adam

    optimizer = opt(params, lr=args.dc_lr, weight_decay=args.dc_weight_decay)
    new_lr = args.dc_lr
    return new_lr, optimizer


def get_eval_epochs_iterations(args, m_loader):
    epoch_iterations = len(m_loader.dataset) // m_loader.batch_size
    eval_frac, eval_intp = math.modf(args.dc_update_interval)
    eval_epochs = int(eval_intp)
    eval_iterations = int(eval_frac * epoch_iterations) + 1  # Round up to avoid eval at last iter
    if eval_epochs == 0:
        eval_epochs = 1  # Eval every epoch
    if eval_iterations == 1:
        eval_iterations = epoch_iterations + 1  # Once every evaluation epoch
    return eval_epochs, eval_iterations


def calc_curr_p(dc_model: DCModel, data_loader: DataLoader, data_ind=0, device='cuda:0'):
    p = []
    y_pred = []
    for it, data_arr in enumerate(tqdm(data_loader, desc="P Calculation")):
        with torch.no_grad():
            pose_data = data_arr[data_ind].to(device)
            curr_q, _ = dc_model(pose_data)
            curr_p = dc_model.target_distribution(curr_q)
            y_pred_curr = torch.argmax(curr_q, 1)
            p.append(curr_p.cpu().numpy())
            y_pred.append(y_pred_curr.cpu().numpy())

    p = np.concatenate(p, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return p, y_pred


def eval_clustering_stop_cret(y_pred, y_pred_prev, stop_cret=1e-3):
    stop_flag = False
    delta_label = np.sum(y_pred != y_pred_prev).astype(np.float32) / y_pred.shape[0]
    print('delta_label ', delta_label)
    y_pred_prev = np.copy(y_pred)
    if delta_label < stop_cret:
        print('delta_label ', delta_label, '< tol ', stop_cret)
        print('Reached tolerance threshold. Stopping training if past min epochs.')
        stop_flag = True
    return stop_flag, y_pred_prev, delta_label


def cluster(dataset, encoder: EncoderWrapper, k, device):
    batch_size = 1024
    data_loader = DataLoader(
        dataset, batch_size, True
    )
    encoded_data = torch.tensor([], device=device)
    encoder.eval()
    with torch.no_grad():
        for it, data in enumerate(data_loader):
            data = data[0]
            hidden_state = encoder(data.to(device)).to(device)
            encoded_data = torch.cat((encoded_data, hidden_state), dim=0)

    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=1000
    )
    encoded_data = encoded_data.to('cpu').detach().numpy()
    print('Initializing cluster centers with k-means...')
    kmeans.fit(encoded_data)
    print('Clustering finished!')
    return kmeans.cluster_centers_