"""Convolution kernel for S4D."""

import torch

from .base import S4DBaseKernel
from .utils import register_param, register_cache
from .discretize import materialize_continuous_params, discretize

class S4DConvKernel(S4DBaseKernel):
    """
    S4D convolution kernel.
    This class generates a convolution kernel from diagonal SSM parameters.
    """
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, **kwargs):
        super().__init__(d_model, N, dt_min, dt_max, lr, **kwargs)

        register_cache(self, 'cached_A', None)
        register_cache(self, 'cached_dt', None)
        register_cache(self, 'cached_C', None)

    def prepare_for_inference(self):
        """Prepare the kernel for inference by caching the discrete parameters."""
        self._build_cache()

    def _build_cache(self):
        A, dt, C = materialize_continuous_params(self)
        self.cached_A = A
        self.cached_dt = dt
        self.cached_C = C

    def forward(self, u, D, dropout, activation, output_linear, transposed=True):
        """
        Convolve input u with the S4D kernel.
        Input and output shape (B, H, L) 
        K = 2 * Re{ C * B_bar * exp(A_bar * t) } where t = [0, 1, ..., L-1]
        y = iFFT(FFT(K * 2L), FFT(u * 2L))_{:L}
        """
        if not transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        # B, H, L = u.shape

        # Materialize parameters
        if self.training:
            A, dt, C = materialize_continuous_params(self)
        else:
            A = self.cached_A
            dt = self.cached_dt
            C = self.cached_C

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A # (H N L)
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        # Convolution
        k_f = torch.fft.rfft(K, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * D.unsqueeze(-1)

        y = dropout(activation(y))
        y = output_linear(y)
        if not transposed: y = y.transpose(-1, -2)

        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
