import logging
import os
from itertools import chain

import hydra
from omegaconf import DictConfig

from factory.epistemic_uncertainty_trainer import UncertaintyTrainer
from epistemic_uncertainty.main import load_dc_model
from data_loader import get_dataloader, get_dataset
from factory.trainer import Trainer
from losses import LOSSES
from models import MODELS
from optimizers import OPTIMIZERS
from path_definition import HYDRA_PATH
from schedulers import SCHEDULERS
from utils.reporter import Reporter
from utils.save_load import load_snapshot, save_snapshot, setup_training_dir
from factory.epistemic_uncertainty_evaluator import UncertaintyEvaluator

logger = logging.getLogger(__name__) 


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def train(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    train_dataloader = get_dataloader(cfg.train_dataset, cfg.data)
    train_dataset = get_dataset(cfg.train_dataset, cfg.data)
    cfg.data.is_testing = True
    valid_dataloader = get_dataloader(cfg.valid_dataset, cfg.data)

    # you either train the pose prediction model or the epu model
    # the if part here is about training a pose prediction model and by default this is active
    # the else part is about training the epu model
    if not cfg.train_epu:
        if cfg.load_path is not None:
            model, loss_module, optimizer, optimizer_args, epoch, train_reporter, valid_reporter = load_snapshot(
                cfg.load_path)
            cfg.start_epoch = epoch
            cfg.optimizer = optimizer_args
            cfg.save_dir = cfg.load_path[:cfg.load_path.rindex('snapshots/')]
        else:
            cfg.model.keypoints_num = train_dataloader.dataset.keypoints_num
            # cfg.model.mean_pose = train_dataloader.dataset.mean_pose
            # cfg.model.std_pose = train_dataloader.dataset.std_pose

            model = MODELS[cfg.model.type](cfg.model)
            
            loss_module = LOSSES[cfg.model.loss.type](cfg.model.loss)
            
            optimizer = OPTIMIZERS[cfg.optimizer.type](
                chain(model.parameters(), loss_module.parameters()), cfg.optimizer)
            
            train_reporter = Reporter(state='train')
            valid_reporter = Reporter(state='valid')
            cfg.save_dir = os.getcwd()
            setup_training_dir(cfg.save_dir)
            save_snapshot(model, loss_module, optimizer, cfg.optimizer,
                        0, train_reporter, valid_reporter, cfg.save_dir)
        scheduler = SCHEDULERS[cfg.scheduler.type](optimizer, cfg.scheduler)
        trainer = Trainer(cfg, train_dataloader, valid_dataloader, model, loss_module, optimizer, cfg.optimizer, scheduler,
                        train_reporter, valid_reporter)
        trainer.train()

    else:
        # this part is for training the epistemic uncertainty model
        uncertainty_model: None
        uncertainty_trainer = UncertaintyTrainer(cfg, train_dataset, train_dataloader, valid_dataloader)
        uncertainty_trainer.train()


if __name__ == '__main__':
    train()
