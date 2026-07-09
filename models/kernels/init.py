"""Initialize S4D parameters."""

import math
import torch
from einops import repeat


def init_s4d_params(d_model, N, dt_min, dt_max, lr):
    """
        Initialize S4D parameters.
        - A = -exp(log_A_real) + 1j * A_imag
        - dt = exp(log_dt)
        - C = complex-valued random variable

        S4d-Lin initialization:
        - A_real = -0.5
        - A_imag = pi * n
    """
    # Generate dt
    H = d_model
    # log_dt: time step for each channel, sampled uniformly in log space
    log_dt = torch.rand(H) * (
        math.log(dt_max) - math.log(dt_min)
    ) + math.log(dt_min)

    # C: complex-valued random variable
    C = torch.randn(H, N // 2, dtype=torch.cfloat)
    
    # A: S4D-lin initialization
    log_A_real = torch.log(0.5 * torch.ones(H, N//2))
    A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)

    return log_dt, C, log_A_real, A_imag