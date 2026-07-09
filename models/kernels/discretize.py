"""discretize.py: Discretization routines for continuous-time SSMs"""

import torch

def materialize_continuous_params(module):
    """
    Materialize continuous-time SSM parameters from their log-space representations.

    A = -exp(log_A_real) + j * A_imag
    dt = exp(log_dt)
    """
    
    dt = torch.exp(module.log_dt) # (H)
    C = torch.view_as_complex(module.C) # (H N)
    A = -torch.exp(module.log_A_real) + 1j * module.A_imag # (H N)
    return A, dt, C

def discretize(A, dt):
    """
    Zero-order hold discretization

    A_bar = exp(A * dt)
    B_bar = (exp(A * dt ) - I ) / A

    Discretize a diagonal SSM with dt using the bilinear transform.
    A: (..., N) complex-valued diagonal SSM
    dt: (...) real-valued time step
    returns: (..., N) complex-valued discrete-time SSM
    """

    dtA = A * dt.unsqueeze(-1)  # (H N)
    A_bar = torch.exp(dtA)  # (H N)
    B_bar = (torch.exp(dtA) - 1.0) / A  # (H N)
    return A_bar, B_bar
