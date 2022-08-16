import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if not m.bias is None:
            nn.init.zeros_(m.bias)
        print("xavier init")

def get_param(shape):
    param = Parameter(torch.Tensor(*shape))
    xavier_normal_(param.data)
    return param

