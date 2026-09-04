import torch.nn as nn
import torch
from nets.normalization import Identity


def get_activation_layer(act_type='elu'):
    if act_type == 'elu':
        return nn.ELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'None':
        return Identity()
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=1, init=0.25)
    else:
        raise
