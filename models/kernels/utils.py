"""utils.py"""

import torch.nn as nn

def register_param(module, name, tensor, lr=None):
    """Register a tensor with a configurable learning rate and 0 weight decay"""

    if lr == 0.0:
        module.register_buffer(name, tensor)
    else:
        module.register_parameter(name, nn.Parameter(tensor))

        optim = {"weight_decay": 0.0}
        if lr is not None: 
            optim["lr"] = lr
        setattr(getattr(module, name), "_optim", optim)

def register_cache(module, name, tensor):
    """Register a tensor as a buffer (non-trainable)"""
    module.register_buffer(name, tensor, persistent=False)