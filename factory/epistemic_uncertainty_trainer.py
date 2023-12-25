import logging
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from epistemic_uncertainty.model.dc.deep_clustering import DCModel
from epistemic_uncertainty.model.dc.train_dc import cluster, train_dc_model
from epistemic_uncertainty.model.lstm.lstm import LstmAutoEncoder, EncoderWrapper
from epistemic_uncertainty.model.lstm.train_lstm import train_lstm_model
from epistemic_uncertainty.utils.dataset_utils import INCLUDED_JOINTS_COUNT
from epistemic_uncertainty.utils.train_utils import save_model

torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)


class UncertaintyTrainer:
    def __init__(self, args, train_dataset, train_dataloader, valid_dataloader):
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.train_dataset = train_dataset

    def train(self):
        logger.info("Training started.")
        time0 = time.time()
        self.__train_dc_model()
        logger.info("-" * 100)
        logger.info('Training is completed in %.2f seconds.' % (time.time() - time0))

    def __train_dc_model(self):
        dev = self.args.device
        n_clusters = self.args.n_clusters
        exp_name = self.args.experiment_name
        lstm_ae = LstmAutoEncoder(pose_dim=INCLUDED_JOINTS_COUNT[self.args.dataset_name], dev=dev)
        train_lstm_model(self.args.lstm, lstm_ae, self.train_dataloader, self.valid_dataloader, dev=dev)
        torch.save(lstm_ae, f'lstm_model_{exp_name}.pt')
        lstm_ae.eval()
        lstm_ae.to(dev)
        encoder = EncoderWrapper(lstm_ae).to(dev)
        initial_clusters = cluster(self.train_dataset, encoder, n_clusters, dev)
        dc_model = DCModel(lstm_ae=lstm_ae, n_clusters=n_clusters,
                           initial_clusters=initial_clusters,
                           device=dev)
        best_model = train_dc_model(self.args.dc, dc_model, self.train_dataset, self.args.batch_size,
                                    num_workers=self.args.num_workers,
                                    dev=dev)
        save_model(dc_model, f'dc_model_{exp_name}')
        save_model(best_model, f'dc_model_best_{exp_name}', best=True)
        return dc_model
