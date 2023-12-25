import torch.nn as nn
import potr.utils as utils

import os, sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

from potr import INIT_FUNC
from potr.pose_gcn import SimpleEncoder

def pose_encoder_gcn(args):
  encoder = SimpleEncoder(
      n_nodes=args.n_major_joints,
      input_features=9 if args.pose_format == 'rotmat' else 3,
      model_dim=args.model_dim, 
      p_dropout=args.dropout
  )

  return encoder


def pose_decoder_mlp(args):
    init_fn = INIT_FUNC[args.init_fn_name]
    pose_decoder = nn.Linear(args.model_dim, args.pose_dim*args.n_major_joints)
    utils.weight_init(pose_decoder, init_fn=init_fn)
    return pose_decoder

def select_pose_encoder_decoder_fn(args):

  if args.pose_embedding_type.lower() == 'gcn_enc':
    return pose_encoder_gcn, pose_decoder_mlp

  else:
    raise ValueError('Unknown pose embedding {}'.format(args.pose_embedding_type))