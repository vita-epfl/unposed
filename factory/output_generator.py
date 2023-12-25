import logging
import time
import pandas as pd
import torch

from utils.save_load import save_test_results
from utils.others import dict_to_device

logger = logging.getLogger(__name__)


class Output_Generator:
    def __init__(self, model, dataloader, save_dir, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.device = device

        self.result = pd.DataFrame()
        self.pred_pose = torch.Tensor().to(device)

    def generate(self):
        logger.info("Generating outputs started.")
        self.model.eval()
        time0 = time.time()
        self.__generate()
        save_test_results(self.result, [self.pred_pose], self.save_dir)
        logger.info('Generating outputs is completed in: %.2f' % (time.time() - time0))

    def __generate(self):
        for data in self.dataloader:
            with torch.no_grad():
                # predict & calculate loss
                model_outputs = self.model(dict_to_device(data, self.device))
                assert 'pred_pose' in model_outputs.keys(), 'outputs of model should include pred_pose'
                pred_pose = model_outputs['pred_pose']

                self.store_results(pred_pose)

    def store_results(self, pred_pose):
        # update tensors
        self.pred_pose = torch.cat((self.pred_pose, pred_pose), 0)

        # to cpu
        if self.device == 'cuda':
            pred_pose = pred_pose.detach().cpu()
        # update dataframe
        for i in range(pred_pose.shape[0]):
            single_data = {'pred_pose': str(pred_pose[i].numpy().tolist())}
            self.result = self.result.append(single_data, ignore_index=True)
