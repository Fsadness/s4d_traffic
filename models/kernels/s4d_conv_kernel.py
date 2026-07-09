"""Convolution kernel for S4D."""

import torch

from .base import S4DBaseKernel
from .discretize import materialize_continuous_params, discretize

class S4DConvKernel(S4DBaseKernel):
    """
    S4D convolution kernel.
    This class generates a convolution kernel from diagonal SSM parameters.
    """
    def forward(self, u):
        """
        Convolve input u with the S4D kernel.
        Input and output shape (B, H, L) 
        K = 2 * Re{ C * B_bar * exp(A_bar * t) } where t = [0, 1, ..., L-1]
        y = iFFT(FFT(K * 2L), FFT(u * 2L))_{:L}
        """
        B, H, L = u.shape

        # Materialize parameters
        A, dt, C = materialize_continuous_params(self)


        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        B_bar = (torch.exp(dtA) - 1.0) / A  # (H N)
        K = 2 * torch.einsum('hn, hnl -> hl', C * B_bar, torch.exp(K)).real

        # Convolution
        k_f = torch.fft.rfft(K, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        return y
