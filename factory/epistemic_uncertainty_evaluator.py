import logging
import time

import torch
import numpy as np
import os
from tqdm import tqdm

from utils.others import dict_to_device
from epistemic_uncertainty.utils.uncertainty import calculate_pose_uncertainty

logger = logging.getLogger(__name__)


class UncertaintyEvaluator:
    def __init__(self, args, dataloader, model, uncertainty_model, reporter):
        self.args = args
        self.dataloader = dataloader
        self.model = model.to(args.device)
        self.uncertainty_model = uncertainty_model
        self.reporter = reporter
        self.rounds_num = args.rounds_num
        self.device = args.device

    def evaluate(self):
        logger.info('Epistemic uncertainty evaluation started.')
        self.model.eval()
        for i in range(self.rounds_num):
            logger.info('round ' + str(i + 1) + '/' + str(self.rounds_num))
            self.__evaluate()
        self.reporter.print_pretty_metrics(logger, ['UNCERTAINTY'])
        self.reporter.save_csv_metrics(['UNCERTAINTY'], os.path.join(self.args.save_dir, "uncertainty_eval.csv"))
        logger.info("Epistemic uncertainty evaluation has been completed.")

    def __evaluate(self):
        self.reporter.start_time = time.time()
        pose_key = None
        for data in tqdm(self.dataloader):
            actions = set(data['action']) if 'action' in data.keys() else set()
            actions.add("all")
            if pose_key is None:
                pose_key = [k for k in data.keys() if "pose" in k][0]
            batch_size = data[pose_key].shape[0]
            with torch.no_grad():
                # calculate uncertainty
                model_outputs = self.model(dict_to_device(data, self.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'

                # calculate pose_metrics
                report_attrs = {}
                dynamic_counts = {}

                pred_metric_pose = model_outputs['pred_pose']

                future_metric_pose = data['future_pose']

                for action in actions:
                    if action == "all":
                        metric_value = calculate_pose_uncertainty(pred_metric_pose.to(self.device),
                                                                  self.uncertainty_model,
                                                                  self.args.dataset_name)
                    else:
                        indexes = np.where(np.asarray(data['action']) == action)[0]
                        metric_value = calculate_pose_uncertainty(pred_metric_pose.to(self.device)[indexes],
                                                                  self.uncertainty_model,
                                                                  self.args.dataset_name)
                        dynamic_counts[f'UNCERTAINTY_{action}'] = len(indexes)
                    report_attrs[f'UNCERTAINTY_{action}'] = metric_value

                self.reporter.update(report_attrs, batch_size, True, dynamic_counts)

        self.reporter.epoch_finished()
