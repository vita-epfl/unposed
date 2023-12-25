import numpy as np
import torch
import cv2

def pose_from_vel(velocity, last_obs_pose, stay_in_frame=False):
    device = 'cuda' if velocity.is_cuda else 'cpu'
    pose = torch.zeros_like(velocity).to(device)
    last_obs_pose_ = last_obs_pose

    for i in range(velocity.shape[-2]):
        pose[..., i, :] = last_obs_pose_ + velocity[..., i, :]
        last_obs_pose_ = pose[..., i, :]

    if stay_in_frame:
        for i in range(velocity.shape[-1]):
            pose[..., i] = torch.min(pose[..., i], 1920 * torch.ones_like(pose.shape[:-1]).to(device))
            pose[..., i] = torch.max(pose[..., i], torch.zeros_like(pose.shape[:-1]).to(device))

    return pose


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def get_binary(src, device):
    zero = torch.zeros_like(src).to(device)
    one = torch.ones_like(src).to(device)
    return torch.where(src > 0.5, one, zero)


def dict_to_device(src, device):
    out = dict()
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone().to(device)
        else:
            out[key] = value
    return out


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)
    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def rotmat_to_expmap(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Rotation matrices for exponenital maps [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmat = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  expmap = np.zeros([n_samples*n_joints, 3, 1])
  for i in range(expmap.shape[0]):
    expmap[i] = cv2.Rodrigues(rotmat[i])[0]
  expmap = np.reshape(expmap, [n_samples, n_joints, 3])
  return expmap

def expmap_to_rotmat(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 3]
  Returns:
    Rotation matrices for exponenital maps [n_samples, n_joints, 9].
  """
  n_samples, n_joints, _ = action_sequence.shape
  expmap = np.reshape(action_sequence, [n_samples*n_joints, 1, 3])
  # first three values are positions, so technically it's meaningless to convert them,
  # but we do it anyway because later we discard this values anywho
  rotmats = np.zeros([n_samples*n_joints, 3, 3])
  for i in range(rotmats.shape[0]):
    rotmats[i] = cv2.Rodrigues(expmap[i])[0]
  rotmats = np.reshape(rotmats, [n_samples, n_joints, 3*3])
  return rotmats

def rotmat_to_euler(action_sequence):
  """Convert exponential maps to rotmats.

  Args:
    action_sequence: [n_samples, n_joints, 9]
  Returns:
    Euler angles for rotation maps given [n_samples, n_joints, 3].
  """
  n_samples, n_joints, _ = action_sequence.shape
  rotmats = np.reshape(action_sequence, [n_samples*n_joints, 3, 3])
  eulers = np.zeros([n_samples*n_joints, 3])
  for i in range(eulers.shape[0]):
    eulers[i] = rotmat2euler(rotmats[i])
  eulers = np.reshape(eulers, [n_samples, n_joints, 3])
  return eulers

def rotmat2euler(R):
  """Converts a rotation matrix to Euler angles.
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

  Args:
    R: a 3x3 rotation matrix

  Returns:
    eul: a 3x1 Euler angle representation of R
  """
  if R[0,2] >= 1 or R[0,2] <= -1:
    # special case values are out of bounds for arcsinc
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;
  else:
    E2 = -np.arcsin(R[0,2])
    E1 = np.arctan2(R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2(R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);
  return eul

def expmap_to_euler(action_sequence):
  rotmats = expmap_to_rotmat(action_sequence)
  eulers = rotmat_to_euler(rotmats)
  return eulers

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def normalize(in_tensor, mean, std):
    device = 'cuda' if in_tensor.is_cuda else 'cpu'
    bs, frame_n, feature_n = in_tensor.shape
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    keypoint_dim = mean.shape[0]
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor - mean) / std


def denormalize(in_tensor, mean, std):
    device = 'cuda' if in_tensor.is_cuda else 'cpu'
    bs, frame_n, feature_n = in_tensor.shape
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    keypoint_dim = mean.shape[0]
    assert mean.shape == std.shape
    assert feature_n % keypoint_dim == 0
    mean = mean.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)
    std = std.unsqueeze(0).repeat(bs, frame_n, feature_n // keypoint_dim)

    return (in_tensor * std) + mean


def xyz_to_spherical(inputs):
    """
    Convert cartesian representation to spherical representation.
    Args:
      inputs -- cartesian coordinates. (..., 3)
    
    Returns:
      out -- spherical coordinate. (..., 3)
    """
    
    rho = torch.norm(inputs, p=2, dim=-1)
    theta = torch.arctan(inputs[..., 2] / (inputs[..., 0] + 1e-8)).unsqueeze(-1)
    tol = 0
    theta[inputs[..., 0] < tol] = theta[inputs[..., 0] < tol] + torch.pi
    phi = torch.arccos(inputs[..., 1] / (rho + 1e-8)).unsqueeze(-1)
    rho = rho.unsqueeze(-1)
    out = torch.cat([rho, theta, phi], dim=-1)
    out[out.isnan()] = 0

    return out

def spherical_to_xyz(self, inputs):
    """
    Convert cartesian representation to spherical representation.
    Args:
      inputs -- spherical coordinates. (..., 3)
    
    Returns:
      out -- cartesian coordinate. (..., 3)
    """
    
    x = inputs[..., 0] * torch.sin(inputs[..., 2]) * torch.cos(inputs[..., 1])
    y = inputs[..., 0] * torch.sin(inputs[..., 2]) * torch.sin(inputs[..., 1])
    z = inputs[..., 0] * torch.cos(inputs[..., 2])
    x, y, z = x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)

    return torch.cat([x, z, y], dim=-1)

def sig5(p:torch.Tensor, x:torch.Tensor):
    """
    Arguments:
        p -- sig5 parameters. shape: ..., 5
        x -- input of sig5 function. shape: ... 
    Return:
        output -- output of sig5 function. 
    """
    assert p.shape[-1] == 5
    if len(p.shape) == 1: p = p.reshape(1, -1)
    p_shape = p.shape 
    x_shape = x.shape 

    p = p.reshape(-1, 5) # 20, 5
    x = x.reshape(1, -1) # 1, 23
    
    p1 = p[:, 0].unsqueeze(1) # 20, 1
    p2 = p[:, 1].unsqueeze(1)
    p3 = p[:, 2].unsqueeze(1)
    p4 = p[:, 3].unsqueeze(1)
    p5 = p[:, 4].unsqueeze(1)

    c = 2*p3*p5/torch.abs(p3+p5) # 20, 1
    f = 1/(1+torch.exp(-c*(p4-x))) # 20, 23
    g = torch.exp(p3*(p4-x)) # 20, 23
    h = torch.exp(p5*(p4-x)) # 20, 23
    output = (p1+(p2/(1+f*g+(1-f)*h))) # 20, 23
    output = output.reshape(*p_shape[:-1], *x_shape)
    return output

def polyx(p:torch.Tensor, input:torch.Tensor, x:int):
    """
    Arguments:
        p -- polyx parameters. shape: ..., 10
        input -- input of polyx function. shape: ... 
        x -- degree of polynomial function.  
    Return:
        output -- output of polyx function. 
    """
    assert p.shape[-1] == x+1
    if len(p.shape) == 1: p = p.reshape(1, -1)
    p_shape = p.shape # ..., x+1
    input_shape = input.shape # ...

    input = input.reshape(1, -1) # ..., 1
    
    powers = torch.arange(x+1).reshape(-1,1).to(input.device) # x+1, 1
    p = p.unsqueeze(-1) # ..., x+1, 1
    print(input.shape, powers.shape, p.shape)
    return (p*(input**powers)).sum(dim=-2).reshape(*p_shape[:-1], *input_shape)

def sigstar(p:torch.Tensor, x:torch.Tensor):
    """
    Arguments:
        p -- sig* parameters. shape: ..., 3
        x -- input of sig* function. shape: ... 
    Return:
        output -- output of sig* function. 
    """
    assert p.shape[-1] == 3
    if len(p.shape) == 1: p = p.reshape(1, -1)
    p_shape = p.shape 
    x_shape = x.shape 

    p = p.reshape(-1, 3) # 20, 3
    x = x.reshape(1, -1) # 1, 23
    
    x0 = p[:, 0].unsqueeze(1) # 20, 1
    k = p[:, 1].unsqueeze(1)
    L = p[:, 2].unsqueeze(1)

    output = L / (1 + torch.exp(-k * (x - x0))) # 20, 23
    output = output.reshape(*p_shape[:-1], *x_shape) # 
    return output


p3d0_base = torch.tensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 7.2556e-02, -9.0371e-02, -4.9508e-03],
         [-7.0992e-02, -8.9911e-02, -4.2638e-03],
         [-2.9258e-03,  1.0815e-01, -2.7961e-02],
         [ 1.1066e-01, -4.7893e-01, -7.1666e-03],
         [-1.1376e-01, -4.8391e-01, -1.1530e-02],
         [ 3.5846e-03,  2.4726e-01, -2.5113e-02],
         [ 9.8395e-02, -8.8787e-01, -5.0576e-02],
         [-9.9592e-02, -8.9208e-01, -5.4003e-02],
         [ 5.3301e-03,  3.0330e-01, -1.3979e-04],
         [ 1.3125e-01, -9.4635e-01,  7.0107e-02],
         [-1.2920e-01, -9.4181e-01,  7.1206e-02],
         [ 2.4758e-03,  5.2506e-01, -3.7885e-02],
         [ 8.6329e-02,  4.2873e-01, -3.4415e-02],
         [-7.7794e-02,  4.2385e-01, -4.0395e-02],
         [ 8.1987e-03,  5.9696e-01,  1.8670e-02],
         [ 1.7923e-01,  4.6251e-01, -4.3923e-02],
         [-1.7389e-01,  4.5846e-01, -5.0048e-02],
         [ 4.4708e-01,  4.4718e-01, -7.2309e-02],
         [-4.3256e-01,  4.4320e-01, -7.3162e-02],
         [ 7.0520e-01,  4.5867e-01, -7.2730e-02],
         [-6.9369e-01,  4.5237e-01, -7.7453e-02]]])

def DPWconvertTo3D(pose_seq):
    res = []
    for pose in pose_seq:
        assert len(pose.shape) == 2 and pose.shape[1] == 72
        
        pose = torch.from_numpy(pose).float()
        pose = pose.view(-1, 72//3, 3)
        pose = pose[:, :-2]
        pose[:, 0] = 0
        res.append(ang2joint(pose).reshape(-1, 22 * 3).detach().numpy())
    return np.array(res)

def AMASSconvertTo3D(pose):
    assert len(pose.shape) == 2 and pose.shape[1] == 156
    pose = torch.from_numpy(pose).float()
    pose = pose.view(-1, 156//3, 3)
    pose[:, 0] = 0
    return ang2joint(pose).reshape(-1, 22 * 3).detach().numpy()

def ang2joint(pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """
    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """

    assert len(pose.shape) == 3 and pose.shape[2] == 3
    batch_num = pose.shape[0]
    p3d0 = p3d0_base.repeat([batch_num, 1, 1])

    jnum = 22

    J = p3d0
    R_cube_big = rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )

    for i in range(1, jnum):
        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed


def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].
    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R


def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
    Parameter:
    ---------
    x: Tensor to be appended.
    Return:
    ------
    Tensor after appending of shape [4,4]
    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]
    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.
    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret

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


if __name__ == '__main__':
    # p = torch.tensor([[1, 1, 1], [1, 2, 3]]) # 2, 3
    # input = torch.tensor([[2, 5, 7], [1, 2, 3]]) # 2, 3
    p = torch.randn(22, 8)
    input = torch.arange(0, 35)
    x = 7
    print(polyx(p, input, x).shape)
    # p = torch.rand(3, 4, 3)
    # x = torch.rand(2, 6, 3, 5)
    # print(sigstar(p, x).shape)
