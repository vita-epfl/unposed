import time
from argparse import Namespace
from functools import partial

import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .lstm import LstmAutoEncoder
from ...utils.train_utils import *


def lstm_train_epoch(lstm_model: LstmAutoEncoder, train_loader: DataLoader, loss_func, alpha, optimizer, dev='cuda'):
    """
    Train epoch while training the LSTM model
    :param lstm_model:
    :param train_loader:
    :param loss_func:
    :param alpha:
    :param optimizer:
    :param dev:
    :return:
    """
    lstm_model.train()
    loss_list = []
    for it, data_arr in enumerate(tqdm(train_loader)):
        data = data_arr[0].to(dev, non_blocking=True)
        output = lstm_model(data).to(dev)
        reconstruction_loss = loss_func(output, data)
        reg_loss = calc_reg_loss(lstm_model)
        loss = reconstruction_loss + 1e-3 * alpha * reg_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
    return loss_list


def lstm_eval_epoch(lstm_model: LstmAutoEncoder, valid_loader: DataLoader, loss_func, alpha, dev='cuda'):
    """
    Evaluation epoch while training the LSTM model
    :param lstm_model:
    :param valid_loader:
    :param loss_func:
    :param alpha:
    :param dev:
    :return:
    """
    lstm_model.eval()
    loss_list = []
    for it, data_arr in enumerate(tqdm(valid_loader)):
        data = data_arr[0].to(dev, non_blocking=True)
        output = lstm_model(data).to(dev)
        reconstruction_loss = loss_func(output, data)
        reg_loss = calc_reg_loss(lstm_model)
        loss = reconstruction_loss + 1e-3 * alpha * reg_loss
        loss_list.append(loss.item())
    return loss_list


def train_lstm_model(lstm_cfg, lstm_model: LstmAutoEncoder, train_loader: DataLoader, valid_loader: DataLoader, dev):
    optimizer = init_optimizer(lstm_model, lstm_cfg.optimizer, lr=lstm_cfg.lr)
    scheduler = init_scheduler(optimizer, lstm_cfg.scheduler, lr=lstm_cfg.lr, epochs=lstm_cfg.epochs)
    lstm_model.train()
    lstm_model.to(dev)
    loss_func = nn.MSELoss()
    for epoch in range(lstm_cfg.epochs):
        ep_start_time = time.time()
        print("Started epoch {}".format(epoch))
        train_loss = lstm_train_epoch(lstm_model, train_loader, loss_func, lstm_cfg.alpha, optimizer, dev)
        new_lr = adjust_lr(epoch, lstm_cfg.lr, lstm_cfg.lr_decay, optimizer, scheduler)
        print('lr: {0:.3e}'.format(new_lr))
        eval_loss = lstm_eval_epoch(lstm_model, valid_loader, loss_func, lstm_cfg.alpha, dev)
        print(f'Epoch {epoch + 1}: Training loss = {np.mean(train_loss)} - Eval loss = {np.mean(eval_loss)}, '
              f'took: {time.time() - ep_start_time} seconds')


def init_optimizer(lstm_model: LstmAutoEncoder, type_str, **kwargs):
    """
    Initializes an optimizer for the given LSTM model.
    :param lstm_model:
    :param type_str:
    :param kwargs:
    :return:
    """
    if type_str.lower() == 'adam':
        optimizer = partial(optim.Adam, **kwargs)
    else:
        return None

    return optimizer(lstm_model.parameters())


def init_scheduler(optimizer, type_str, lr, epochs):
    """
    Initializes a scheduler for the given optimizer
    :param optimizer:
    :param type_str:
    :param lr:
    :param epochs:
    :return:
    """
    scheduler = None
    if (type_str.lower() == 'tri') and (epochs >= 8):
        scheduler = partial(optim.lr_scheduler.CyclicLR,
                            base_lr=lr / 10, max_lr=lr * 10,
                            step_size_up=epochs // 8,
                            mode='triangular2',
                            cycle_momentum=False)
    else:
        print("Unable to initialize scheduler, defaulting to exp_decay")

    return scheduler(optimizer)
