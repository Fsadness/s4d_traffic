"""Recurrent S4D kernel implementation."""

import torch

from .base import S4DBaseKernel
from .discretize import materialize_continuous_params, discretize


class S4DRecurrentKernel(S4DBaseKernel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Recurrent state
        self._state = None

    def _get_discrete_params(self):
        """Get discrete-time SSM parameters."""
        A, dt, C = materialize_continuous_params(self)
        A_bar, B_bar = discretize(A, dt)
        return A_bar, B_bar, C

    def reset_state(self, batch_size, device):
        """Reset the recurrent state."""
        if device is None:
            device = next(self.parameters()).device
        self._state = torch.zeros(
            batch_size, self.H, self.N // 2,
            dtype=torch.cfloat, device=device
        )

    def forward_step(self, u_t):
        """
        Perform a single recurrent step of the S4D kernel.
        Input and output shape (B, H)

        x[n] = A_bar * x[n-1] + B_bar * u[n]
        y[n] = Re{2 * sum(C * x[n])}
        """
        if self._state is None:
            raise ValueError("State is not initialized. Call reset_state() first.")

        A_bar, B_bar, C = self._get_discrete_params()

        # Update state
        # x = A_bar * x + B_bar * u_t
        # A_bar, B_bar: (H, N/2) -> (1, H, N/2)
        # u_t: (B, H) -> (B, H, 1)
        self._state = A_bar.unsqueeze(0) * self._state + \
                      B_bar.unsqueeze(0) * u_t.unsqueeze(-1)

        # Output y = Re{2 * sum(C * x)}
        y_t = 2 * (C.unsqueeze(0) * self._state).sum(dim=-1).real

        return y_t

    def forward(self, u):
        """
        Perform recurrent processing of the input sequence u.
        Input shape and output shape (B, H, L)
        """
        B, H, L = u.shape

        A_bar, B_bar, C = self._get_discrete_params()

        # Initialize state
        x = torch.zeros(B, H, self.N // 2, dtype=A_bar.dtype, device=u.device)

        outputs = []
        for t in range(L):
            u_t = u[:, :, t]  # (B, H)
            # Update state
            x = A_bar.unsqueeze(0) * x + B_bar.unsqueeze(0) * u_t.unsqueeze(-1)
            # Output y_t = Re{2 * sum(C * x)}
            y_t = 2 * (C.unsqueeze(0) * x).sum(dim=-1).real
            outputs.append(y_t)

        return torch.stack(outputs, dim=-1) # (B, H, L)