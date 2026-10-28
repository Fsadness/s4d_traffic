"""Recurrent S4D kernel implementation."""

import torch

from .base import S4DBaseKernel
from .utils import register_param, register_cache
from .discretize import materialize_continuous_params, discretize


class S4DRecurrentKernel(S4DBaseKernel):

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None, **kwargs):
        super().__init__(d_model, N, dt_min, dt_max, lr, **kwargs)

        # Recurrent state
        self._state = None

        register_cache(self, 'cached_A_bar', None)
        register_cache(self, 'cached_B_bar', None)
        register_cache(self, 'cached_C', None)

    def prepare_for_inference(self):
        """Prepare the kernel for inference by caching the discrete parameters."""
        self._build_cache()

    def _get_discrete_params(self):
        """Get discrete-time SSM parameters."""
        A, dt, C = materialize_continuous_params(self)
        A_bar, B_bar = discretize(A, dt)
        return A_bar, B_bar, C

    def _build_cache(self):
        A_bar, B_bar, C = self._get_discrete_params()
        self.cached_A_bar = A_bar.unsqueeze(0)  # (1, H, N/2)
        self.cached_B_bar = B_bar.unsqueeze(0)  # (1, H, N/2)
        self.cached_C = C.unsqueeze(0)  # (1, H, N/2)

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
        

        # Update state
        # x = A_bar * x + B_bar * u_t
        # A_bar, B_bar: (H, N/2) -> (1, H, N/2)
        # u_t: (B, H) -> (B, H, 1)
        # self._state = self.cached_A_bar * self._state + self.cached_B_bar * u_t.unsqueeze(-1)
        torch.cuda.nvtx.range_push("state_update")
        self._state.mul_(self.cached_A_bar).addcmul_(self.cached_B_bar, u_t.unsqueeze(-1))
        torch.cuda.nvtx.range_pop()

        # Output y = Re{2 * sum(C * x)}
        y_t = 2 * (self.cached_C * self._state).sum(dim=-1).real # (B, H)

        return y_t

    def forward(self, u, D, dropout, activation, output_linear, transposed=True):
        """
        Perform recurrent processing of the input sequence u.
        Input shape and output shape (B, H, L)
        """
        if not transposed: u = u.transpose(-1, -2)
        # L = u.size(-1)
        B, H, L = u.shape

        if self.training:
            A_bar, B_bar, C = self._get_discrete_params()
            A_bar = A_bar.unsqueeze(0)  # (1, H, N/2)
            B_bar = B_bar.unsqueeze(0)  # (1, H, N/2)
            C = C.unsqueeze(0)  # (1, H, N/2)
        else:
            A_bar = self.cached_A_bar
            B_bar = self.cached_B_bar
            C = self.cached_C

        # Initialize state
        x = torch.zeros(B, H, self.N // 2, dtype=A_bar.dtype, device=u.device)

        outputs = torch.empty(B, H, L, dtype=u.dtype, device=u.device)
        for t in range(L):
            u_t = u[:, :, t]  # (B, H)
            # Update state
            # x = A_bar * x + B_bar * u_t.unsqueeze(-1)
            x.mul_(A_bar).addcmul_(B_bar, u_t.unsqueeze(-1))
            # Output y_t = Re{2 * sum(C * x)}
            outputs[:,:,t] = 2 * (C * x).sum(dim=-1).real

        # Compute D term in state space equation - essentially a skip connection
        y = outputs + u * D.unsqueeze(-1)

        y = dropout(activation(y))
        y = output_linear(y)
        if not transposed: y = y.transpose(-1, -2)

        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
