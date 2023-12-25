import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def ADE(pred, target, dim):
    """
    Average Displacement Error
    """
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., d] - target[..., d]) ** 2
    
    ade = torch.mean(torch.sqrt(displacement)) 
    return ade


def FDE(pred, target, dim):
    """
    Final Displacement Error
    """
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., -1, :, d] - target[..., -1, :, d]) ** 2
    fde = torch.mean(torch.sqrt(displacement))
    return fde

def local_ade(pred, target, dim):
    bs, frames, feat = pred.shape
    keypoints = feat // dim
    pred_pose = pred.reshape(bs, frames, keypoints, dim)
    local_pred_pose = pred_pose - pred_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    target_pose = target.reshape(bs, frames, keypoints, dim)
    local_target_pose = target_pose - target_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    local_pred_pose = local_pred_pose.reshape(bs, frames, feat)
    local_target_pose = local_target_pose.reshape(bs, frames, feat)
    return ADE(local_pred_pose, local_target_pose, dim)



def local_fde(pred, target, dim):
    bs, frames, feat = pred.shape
    keypoints = feat // dim
    pred_pose = pred.reshape(bs, frames, keypoints, dim)
    local_pred_pose = pred_pose - pred_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    target_pose = target.reshape(bs, frames, keypoints, dim)
    local_target_pose = target_pose - target_pose[:, :, 0:1, :].repeat(1, 1, keypoints, 1)
    local_pred_pose = local_pred_pose.reshape(bs, frames, feat)
    local_target_pose = local_target_pose.reshape(bs, frames, feat)
    return FDE(local_pred_pose, local_target_pose, dim)

def MSE(pred, target, dim=None):
    """
    Mean Squared Error
    Arguments:
        pred -- predicted sequence : (batch_size, sequence_length, pose_dim*n_joints)

    """
    # target = target.reshape(*target.shape[:-2], -1)
    assert pred.shape == target.shape
    B, S, D = pred.shape
    mean_errors = torch.zeros((B, S))

    # Training is done in exponential map or rotation matrix or quaternion
    # but the error is reported in Euler angles, as in previous work [3,4,5] 
    for i in np.arange(B):
        # seq_len x complete_pose_dim (H36M==99)
        eulerchannels_pred = pred[i] #.numpy()
        # n_seeds x seq_len x complete_pose_dim (H36M==96)
        action_gt = target#srnn_gts_euler[action]
        
        # seq_len x complete_pose_dim (H36M==96)
        gt_i = action_gt[i]#np.copy(action_gt.squeeze()[i].numpy())
        # Only remove global rotation. Global translation was removed before
        gt_i[:, 0:3] = 0

        # here [2,4,5] remove data based on the std of the batch THIS IS WEIRD!
        # (seq_len, 96) - (seq_len, 96)
        idx_to_use = np.where(np.std(gt_i.detach().cpu().numpy(), 0) > 1e-4)[0]
        euc_error = torch.pow(gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)

        euc_error = torch.sum(euc_error, 1)

        euc_error = torch.sqrt(euc_error)
        mean_errors[i,:] = euc_error

    mean_mean_errors = torch.mean(mean_errors, 0)
    return mean_mean_errors.mean()

def VIM(pred, target, dim, mask):
    """
    Visibilty Ignored Metric
    Inputs:
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        target: Ground truth data - array of shape (pred_len, #joint*(2D/3D))
        dim: dimension of data (2D/3D)
        mask: Visibility mask of pos - array of shape (pred_len, #joint)
    Output:
        errorPose:
    """
    assert mask is not None, 'pred_mask should not be None.'

    target_i_global = np.copy(target)
    if dim == 2:
        mask = np.repeat(mask, 2, axis=-1)
        errorPose = np.power(target_i_global - pred, 2) * mask
        # get sum on joints and remove the effect of missing joints by averaging on visible joints
        errorPose = np.sqrt(np.divide(np.sum(errorPose, 1), np.sum(mask, axis=1)))
        where_are_NaNs = np.isnan(errorPose)
        errorPose[where_are_NaNs] = 0
    elif dim == 3:
        errorPose = np.power(target_i_global - pred, 2)
        errorPose = np.sum(errorPose, 1)
        errorPose = np.sqrt(errorPose)
    else:
        msg = "Dimension of data must be either 2D or 3D."
        logger.error(msg=msg)
        raise Exception(msg)
    return errorPose


def VAM(pred, target, dim, mask, occ_cutoff=100):
    """
    Visibility Aware Metric
    Inputs:
        pred: Prediction data - array of shape (pred_len, #joint*(2D/3D))
        target: ground truth data - array of shape (pred_len, #joint*(2D/3D))
        dim: dimension of data (2D/3D)
        mask: Predicted visibilities of pose, array of shape (pred_len, #joint)
        occ_cutoff: Maximum error penalty
    Output:
        seq_err:
    """
    assert mask is not None, 'pred_mask should not be None.'
    assert dim == 2 or dim == 3

    pred_mask = np.repeat(mask, 2, axis=-1)
    seq_err = []
    if type(target) is list:
        target = np.array(target)
    target_mask = np.where(abs(target) < 0.5, 0, 1)
    for frame in range(target.shape[0]):
        f_err = 0
        N = 0
        for j in range(0, target.shape[1], 2):
            if target_mask[frame][j] == 0:
                if pred_mask[frame][j] == 0:
                    dist = 0
                elif pred_mask[frame][j] == 1:
                    dist = occ_cutoff
                    N += 1
            elif target_mask[frame][j] > 0:
                N += 1
                if pred_mask[frame][j] == 0:
                    dist = occ_cutoff
                elif pred_mask[frame][j] == 1:
                    d = np.power(target[frame][j:j + 2] - pred[frame][j:j + 2], 2)
                    d = np.sum(np.sqrt(d))
                    dist = min(occ_cutoff, d)
            else:
                msg = "Target mask must be positive values."
                logger.error(msg)
                raise Exception(msg)
            f_err += dist
        if N > 0:
            seq_err.append(f_err / N)
        else:
            seq_err.append(f_err)
    return np.array(seq_err)


#new:
def F1(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 1, :, d] - target[..., 1, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de

def F3(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 3, :, d] - target[..., 3, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de

def F7(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 7, :, d] - target[..., 7, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de

def F9(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 9, :, d] - target[..., 9, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de

def F13(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 13, :, d] - target[..., 13, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de

def F17(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 17, :, d] - target[..., 17, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de

def F21(pred, target, dim):
    keypoints_num = int(pred.shape[-1] / dim)
    pred = torch.reshape(pred, pred.shape[:-1] + (keypoints_num, dim))
    target = torch.reshape(target, target.shape[:-1] + (keypoints_num, dim))
    displacement = 0
    for d in range(dim):
        displacement += (pred[..., 21, :, d] - target[..., 21, :, d]) ** 2
    de = torch.mean(torch.sqrt(displacement))
    return de