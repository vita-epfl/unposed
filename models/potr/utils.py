import torch.nn as nn
import numpy as np

def xavier_init(layer, mean_, sd_, bias, norm_bias=True):
  classname = layer.__class__.__name__
  if classname.find('Linear')!=-1:
    print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
    nn.init.xavier_uniform_(layer.weight.data)
    # nn.init.xavier_normal(layer.bias.data)
    if norm_bias:
      layer.bias.data.normal_(0, 0.05)
    else:
      layer.bias.data.zero_()

def normal_init(layer, mean_, sd_, bias, norm_bias=True):
  """Intialization of layers with normal distribution with mean and bias"""
  classname = layer.__class__.__name__
  # Only use the convolutional layers of the module
  if classname.find('Linear') != -1:
    print('[INFO] (normal_init) Initializing layer {}'.format(classname))
    layer.weight.data.normal_(mean_, sd_)
    if norm_bias:
      layer.bias.data.normal_(bias, 0.05)
    else:
      layer.bias.data.fill_(bias)

def weight_init(
    module, 
    mean_=0, 
    sd_=0.004, 
    bias=0.0, 
    norm_bias=False, 
    init_fn=normal_init):
  """Initialization of layers with normal distribution"""
  moduleclass = module.__class__.__name__
  try:
    for layer in module:
      if layer.__class__.__name__ == 'Sequential':
        for l in layer:
          init_fn(l, mean_, sd_, bias, norm_bias)
      else:
        init_fn(layer, mean_, sd_, bias, norm_bias)
  except TypeError:
    init_fn(module, mean_, sd_, bias, norm_bias)


def create_look_ahead_mask(seq_length, is_nonautoregressive=False):
  """Generates a binary mask to prevent to use future context in a sequence."""
  if is_nonautoregressive:
    return np.zeros((seq_length, seq_length), dtype=np.float32)
  x = np.ones((seq_length, seq_length), dtype=np.float32)
  mask = np.triu(x, 1).astype(np.float32)
  return mask  # (seq_len, seq_len)


