import torch
from torch import nn
import numpy as np
from utils.others import rotmat_to_euler, expmap_to_rotmat
_MAJOR_JOINTS = [
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 24, 25, 26, 27
]

ACTIONS = {
  "walking": 0, 
  "eating": 1, 
  "smoking": 2, 
  "discussion": 3, 
  "directions": 4,
  "greeting": 5, 
  "phoning": 6, 
  "posing": 7, 
  "purchases": 8, 
  "sitting": 9,
  "sittingdown": 10, 
  "takingphoto": 11, 
  "waiting": 12, 
  "walkingdog": 13,
  "walkingtogether": 14
}

def compute_difference_matrix(self, src_seq, tgt_seq):
    """Computes a matrix of euclidean difference between sequences.

    Args:
      src_seq: Numpy array of shape [src_len, dim].
      tgt_seq: Numpy array of shape [tgt_len, dim].

    Returns:
      A matrix of shape [src_len, tgt_len] with euclidean distances.
    """
    B = src_seq.shape[0]
    src_len = src_seq.shape[1] # M
    tgt_len = tgt_seq.shape[1] # N

    distance = np.zeros((B, src_len, tgt_len), dtype=np.float32)
    for b in range(B):
        for i in range(src_len):
            for j in range(tgt_len):
                distance[b, i, j] = np.linalg.norm(src_seq[b, i]-tgt_seq[b, j])

    row_sums = distance.sum(axis=2)
    distance_norm = distance / row_sums[:, :, np.newaxis]
    distance_norm = 1.0 - distance_norm

    return distance, distance_norm

def train_preprocess(inputs, args):
    
    obs_pose = inputs['observed_pose']
    future_pose = inputs['future_pose']

    obs_pose = obs_pose.reshape(obs_pose.shape[0], obs_pose.shape[1], -1, args.pose_dim)
    future_pose = future_pose.reshape(future_pose.shape[0], future_pose.shape[1], -1, args.pose_dim)

    n_major_joints = len(_MAJOR_JOINTS)
    obs_pose = obs_pose[:, :, _MAJOR_JOINTS]
    future_pose = future_pose[:, :, _MAJOR_JOINTS]
    
    obs_pose = obs_pose.reshape(*obs_pose.shape[:2], -1)
    future_pose = future_pose.reshape(*future_pose.shape[:2], -1)
    
    src_seq_len = args.obs_frames_num - 1
    
    if args.include_last_obs:
      src_seq_len += 1
    
    encoder_inputs = np.zeros((obs_pose.shape[0], src_seq_len, args.pose_dim * n_major_joints), dtype=np.float32)
    decoder_inputs = np.zeros((obs_pose.shape[0], args.future_frames_num, args.pose_dim * n_major_joints), dtype=np.float32)
    decoder_outputs = np.zeros((obs_pose.shape[0], args.future_frames_num, args.pose_dim * n_major_joints), dtype=np.float32)
    
    data_sel = torch.cat((obs_pose, future_pose), dim=1)

    encoder_inputs[:, :, 0:args.pose_dim * n_major_joints] = data_sel[:, 0:src_seq_len,:].cpu()
    decoder_inputs[:, :, 0:args.pose_dim * n_major_joints] = \
        data_sel[:, src_seq_len:src_seq_len + args.future_frames_num, :].cpu()

    decoder_outputs[:, :, 0:args.pose_dim * n_major_joints] = data_sel[:, args.obs_frames_num:, 0:args.pose_dim * n_major_joints].cpu()

    if args.pad_decoder_inputs:
      query = decoder_inputs[:, 0:1, :]
      decoder_inputs = np.repeat(query, args.future_frames_num, axis=1)

    
    model_outputs = {
      'encoder_inputs': torch.tensor(encoder_inputs).reshape((*encoder_inputs.shape[:-1], args.n_major_joints, args.pose_dim)).to(args.device), 
      'decoder_inputs': torch.tensor(decoder_inputs).reshape((*decoder_inputs.shape[:-1], args.n_major_joints, args.pose_dim)).to(args.device), 
      'decoder_outputs': torch.tensor(decoder_outputs).reshape((*decoder_outputs.shape[:-1], args.n_major_joints, args.pose_dim)).to(args.device)
    }
    if args.predict_activity:
      model_outputs['action_ids'] = torch.tensor([ACTIONS[a] for a in inputs['action']]).to(args.device)

    return model_outputs
    

def convert_to_euler(action_sequence_, n_major_joints, pose_format, is_normalized=True):
  """Convert the input exponential maps to euler angles.

  Args:
    action_sequence: Pose exponential maps [batch_size, sequence_length, pose_size].
      The input should not contain the one hot encoding in the vector.
  """
  B, S, D = action_sequence_.shape
  rotmats = action_sequence_.reshape((B*S, n_major_joints, -1))
  if pose_format == 'expmap':
    rotmats = expmap_to_rotmat(rotmats)

  euler_maps = rotmat_to_euler(rotmats)
  
  euler_maps = euler_maps.reshape((B, S, -1))
  return euler_maps

def post_process_to_euler(norm_seq, n_major_joints, n_h36m_joints, pose_format):

    batch_size, seq_length, n_major_joints, pose_dim = norm_seq.shape
    norm_seq = norm_seq.reshape(batch_size, seq_length, n_major_joints*pose_dim)
    euler_seq = convert_to_euler(norm_seq, n_major_joints, pose_format)
    euler_seq = euler_seq.reshape((batch_size, seq_length, n_major_joints, 3))
    p_euler_padded = np.zeros([batch_size, seq_length, n_h36m_joints, 3])
    p_euler_padded[:, :, _MAJOR_JOINTS] = euler_seq
    p_euler_padded = np.reshape(p_euler_padded, [batch_size, seq_length, -1])
    return p_euler_padded