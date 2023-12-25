import numpy as np
import torch as torch

from ..model.sts_gcn.utils import forward_kinematics
from .dataset_utils import JOINTS_TO_IGNORE, JOINTS_EQUAL, JOINTS_TO_INCLUDE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def scale(x, scale_ratio):
    """
    Scales pose data with an arbitrary ratio
    :param x:
    :param scale_ratio:
    :return:
    """
    x = x * scale_ratio
    return x


def rescale_to_original_joints_count(sequence, ground_truth, dataset_name: str):
    """
    Adds ignored joints and reconstructs a pose sequence with 32x3 joints for H36M (22 * 3 for AMASS and 3DPW)
    :param sequence:
    :param ground_truth:
    :param dataset_name:
    :return:
    """
    all_joints_seq = ground_truth.clone()
    all_joints_seq[:, :, JOINTS_TO_INCLUDE[dataset_name]] = sequence
    if dataset_name in JOINTS_TO_IGNORE:
        all_joints_seq[:, :, JOINTS_TO_IGNORE[dataset_name]] = all_joints_seq[:, :, JOINTS_EQUAL[dataset_name]]
    return all_joints_seq


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def find_indices_256(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 128):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2
