import os
import logging
import hydra
from omegaconf import DictConfig

from data_loader import get_dataloader
from models import MODELS
from factory.output_generator import Output_Generator
from utils.save_load import load_snapshot, setup_testing_dir

from path_definition import HYDRA_PATH

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="generate_output")
def generate_output(cfg: DictConfig):
    if cfg.load_path is None and cfg.model is None:
        msg = 'either specify a load_path or config a model.'
        logger.error(msg)
        raise Exception(msg)

    dataloader = get_dataloader(cfg.dataset, cfg.data)

    if cfg.load_path is not None:
        model, _, _, _, _, _, _ = load_snapshot(cfg.load_path)
    else:
        cfg.model.keypoints_num = dataloader.dataset.keypoints_num
        model = MODELS[cfg.model.type](cfg.model)
        if cfg.model.type == 'nearest_neighbor':
            model.train_dataloader = get_dataloader(
                cfg.model.train_dataset, cfg.data)
    cfg.save_dir = os.getcwd()
    setup_testing_dir(cfg.save_dir)

    output_enerator = Output_Generator(
        model, dataloader, cfg.save_dir, cfg.device)
    output_enerator.generate()


if __name__ == '__main__':
    generate_output()