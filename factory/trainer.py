import logging
import time
import torch
import gc
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from path_definition import *
from metrics import POSE_METRICS
from utils.others import dict_to_device
from utils.reporter import Reporter
from utils.save_load import save_snapshot
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)
import mlflow
import mlflow.pytorch
from path_definition import *
from os.path import join


class Trainer:
    def __init__(self, args, train_dataloader, valid_dataloader, model, loss_module, optimizer, optimizer_args,
                 scheduler, train_reporter, valid_reporter, train_uncertainty_evaluator=None, validation_uncertainty_evaluator=None):
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model.to(args.device)
        self.loss_module = loss_module.to(args.device)
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler = scheduler
        self.train_reporter = train_reporter
        self.valid_reporter = valid_reporter
        self.tensor_board = SummaryWriter(args.save_dir)
        self.use_validation = False if valid_dataloader is None else True
        # self.uncertainty_evaluator = uncertainty_evaluator

        self.train_uncertainty_evaluator = train_uncertainty_evaluator
        self.validation_uncertainty_evaluator = validation_uncertainty_evaluator

        mlflow.set_tracking_uri(join(args.mlflow_tracking_uri, 'mlruns') if args.mlflow_tracking_uri else join(ROOT_DIR, 'mlruns'))
        mlflow.set_experiment(args.experiment_name if args.experiment_name else args.model.type)
           
        self.run = mlflow.start_run()

        config_path = os.path.join(os.getcwd(), '.hydra', 'config.yaml')
        mlflow.log_artifact(config_path)

        params = {
            'model': args.model.type,
            **dict(args.model),
            'optimizer': args.optimizer.type,
            **dict(args.optimizer),
            'loss': args.model.loss.type,
            **dict(args.model.loss),
            'scheduler': args.scheduler.type,
            **dict(args.scheduler),
            'obs_frames_num': args.obs_frames_num,
            'pred_frames_num': args.pred_frames_num,
            'tag': args.experiment_tag,
            **dict(args.data),
            'save_dir': args.save_dir
        }
        del params['type']

        mlflow.log_params(params)

    def train(self):
        logger.info("Training started.")
        time0 = time.time()
        self.best_loss = np.inf
        self.best_epoch = -1
        for epoch in range(self.args.start_epoch, self.args.epochs):
            print("epoch:", epoch) #new
            self.__train()
            if self.use_validation:
                self.__validate()
                self.scheduler.step(self.valid_reporter.history['loss'][-1])
            
                if self.best_model:
                    save_snapshot(self.model, self.loss_module, self.optimizer, self.optimizer_args, epoch + 1,
                                self.train_reporter,
                                self.valid_reporter, self.args.save_dir, best_model=True)
                    self.best_model = False
            if self.validation_uncertainty_evaluator is not None:
                self.__validate_uncertainty(train=False)

            if (epoch + 1) % self.args.snapshot_interval == 0 or (epoch + 1) == self.args.epochs:
                save_snapshot(self.model, self.loss_module, self.optimizer, self.optimizer_args, epoch + 1,
                              self.train_reporter,
                              self.valid_reporter, self.args.save_dir)
                self.train_reporter.save_data(self.args.save_dir)
                if self.use_validation:
                    self.valid_reporter.save_data(self.args.save_dir)
                Reporter.save_plots(self.args.save_dir, self.train_reporter.history,
                                    self.valid_reporter.history, self.use_validation)
            # if self.use_validation and
        self.tensor_board.close()
        mlflow.end_run()
        logger.info("-" * 100)
        logger.info('Training is completed in %.2f seconds.' % (time.time() - time0))

    def __train(self):
        self.model.train()
        self.train_reporter.start_time = time.time()
        pose_key = None
        for data in tqdm(self.train_dataloader):
            gc.collect()
            # TODO: fix later
            batch_size = data['observed_pose'].shape[0]
            data = dict_to_device(data, self.args.device)
            # predict & calculate loss
            self.model.zero_grad()
            self.loss_module.zero_grad()
            
            model_outputs = self.model(data)
            loss_outputs = self.loss_module(model_outputs, data)

            assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
            assert 'loss' in loss_outputs.keys(), 'outputs of loss should include loss'

            # backpropagate and optimize

            loss = loss_outputs['loss']
            loss.backward()

            # print(model.para)

            if self.args.optimizer.type == 'sam':
                self.optimizer.first_step(zero_grad=True)

                model_outputs = self.model(data)
                loss_outputs = self.loss_module(model_outputs, data)
                loss = loss_outputs['loss']
                loss.backward()
                self.optimizer.second_step(zero_grad=True)

            else:
                self.optimizer.step()

            loss_outputs['loss'] = loss_outputs['loss'].detach().item()

            # calculate pose_metrics

            report_attrs = loss_outputs
            for metric_name in self.args.pose_metrics:
                metric_func = POSE_METRICS[metric_name]

                pred_metric_pose = model_outputs['pred_pose']
                if 'pred_metric_pose' in model_outputs:
                    pred_metric_pose = model_outputs['pred_metric_pose']

                # TODO: write write a warning =D

                future_metric_pose = data['future_pose']
                if 'future_metric_pose' in data:
                    future_metric_pose = data['future_metric_pose']
                metric_value = metric_func(
                    pred_metric_pose.to(self.args.device),
                    future_metric_pose.to(self.args.device),
                    self.model.args.keypoint_dim
                )

                report_attrs[metric_name] = metric_value.detach().item()

            self.train_reporter.update(report_attrs, batch_size)

        self.train_reporter.epoch_finished(self.tensor_board, mlflow)
        self.train_reporter.print_values(logger)
        # self.train_reporter.print_values(logger, self.model.args.use_mask)
        if self.train_uncertainty_evaluator is not None:
            self.__validate_uncertainty(train=True)

    def __validate(self):
        self.model.eval()
        self.loss_module.eval()

        self.valid_reporter.start_time = time.time()
        pose_key = None
        epoch_loss = 0.0
        for data in tqdm(self.valid_dataloader):
            data = dict_to_device(data, self.args.device)
            batch_size = data['observed_pose'].shape[0]

            with torch.no_grad():
                # predict & calculate loss
                model_outputs = dict_to_device(self.model(data), self.args.device)
                loss_outputs = self.loss_module(model_outputs, dict_to_device(data, self.args.device))
                epoch_loss += loss_outputs['loss'].item()
                
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'

                # calculate pose_metrics
                report_attrs = loss_outputs
                for metric_name in self.args.pose_metrics:
                    metric_func = POSE_METRICS[metric_name]

                    pred_metric_pose = model_outputs['pred_pose']
                    if 'pred_metric_pose' in model_outputs:
                        pred_metric_pose = model_outputs['pred_metric_pose']

                    future_metric_pose = data['future_pose']
                    if 'future_metric_pose' in data:
                        future_metric_pose = data['future_metric_pose']
                    metric_value = metric_func(
                        pred_metric_pose.to(self.args.device),
                        future_metric_pose.to(self.args.device),
                        self.model.args.keypoint_dim
                    )
                    report_attrs[metric_name] = metric_value

                self.valid_reporter.update(report_attrs, batch_size)

        if epoch_loss < self.best_loss:
            self.best_model = True
            self.best_loss = epoch_loss

        self.valid_reporter.epoch_finished(self.tensor_board, mlflow)
        self.valid_reporter.print_values(logger)

    def __validate_uncertainty(self, train=True):
        if train:
            self.train_uncertainty_evaluator.evaluate()
        else:
            self.validation_uncertainty_evaluator.evaluate()
