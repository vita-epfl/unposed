import logging

import hydra
from omegaconf import DictConfig

from path_definition import HYDRA_PATH
from preprocessor.dpw_preprocessor import Preprocessor3DPW
from preprocessor.human36m_preprocessor import Human36mPreprocessor
from preprocessor.amass_preprocessor import AmassPreprocessor
from data_loader import DATASETS, DATA_TYPES

logger = logging.getLogger(__name__)


@hydra.main(config_path=HYDRA_PATH, config_name="preprocess")
def preprocess(cfg: DictConfig):
    assert cfg.dataset in DATASETS, "invalid dataset name"
    assert cfg.data_type in DATA_TYPES, "data_type choices: " + str(DATA_TYPES)

    if cfg.dataset == 'human3.6m':
        preprocessor = Human36mPreprocessor(
            dataset_path=cfg.annotated_data_path,
            custom_name=cfg.output_name, 
        )
    elif cfg.dataset == '3dpw':
        preprocessor = Preprocessor3DPW(
            dataset_path=cfg.annotated_data_path,
            custom_name=cfg.output_name, 
            load_60Hz=cfg.load_60Hz
        )
    elif cfg.dataset == 'amass':
        preprocessor = AmassPreprocessor(
            dataset_path=cfg.annotated_data_path,
            custom_name=cfg.output_name, 
        )
    else:
        msg = "Invalid preprocessor."
        logger.error(msg)
        raise Exception(msg)
    preprocessor.normal(data_type=cfg.data_type)


if __name__ == '__main__':
    preprocess()