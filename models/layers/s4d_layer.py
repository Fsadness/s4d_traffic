

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd

from models.kernels import S4DConvKernel, S4DRecurrentKernel

class S4DLayer(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, kernel_type='conv', **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
    
        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        if kernel_type == 'conv':
            self.kernel = S4DConvKernel(self.h, N=self.n, **kernel_args)
        elif kernel_type == 'recurrent':
            self.kernel = S4DRecurrentKernel(self.h, N=self.n, **kernel_args)
            self.reset_state = self.kernel.reset_state
        else:
            raise ValueError(f'Unknown kernel type: {kernel_type}')
        
        self.kernel_type = kernel_type

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward_step(self, u_t):
        """ Forward step for recurrent kernel. Input and output shape (B, H) """
        if self.kernel_type != 'recurrent':
            raise ValueError('forward_step is only available for recurrent kernel.')
        
        y_t = self.kernel.forward_step(u_t) # (B, H)

        # Compute D term in state space equation - essentially a skip connection
        y_t = y_t + u_t * self.D # (B, H)

        y_t = self.dropout(self.activation(y_t))
        # Output linear expects (B, H, L) so we add a dummy length dimension
        y_t = y_t.unsqueeze(-1) # (B, H, 1)
        y_t = self.output_linear(y_t).squeeze(-1)

        return y_t

    def forward(self, u):
        """Full forward pass. Input and output shape (B, H, L)"""
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        y = self.kernel(u) # (B, H, L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
