"""Base kernel implementations."""

import torch
import torch.nn as nn

from .init import init_s4d_params
from .utils import register_param

class S4DBaseKernel(nn.Module):
    """
    base class for S4D Kernels. 
    This class is not meant to be used directly, 
    but rather to be subclassed by specific implementations of S4D kernels.
    """
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()

        self.H = d_model
        self.N = N

        log_dt, C, log_A_real, A_imag = init_s4d_params(d_model, N, dt_min, dt_max, lr)

        register_param(self, 'log_dt', log_dt, lr)
        register_param(self, 'log_A_real', log_A_real, lr)
        register_param(self, 'A_imag', A_imag, lr)

        self.C = nn.Parameter(torch.view_as_real(C))