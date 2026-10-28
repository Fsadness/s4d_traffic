

import torch
import torch.nn as nn

from models.layers.s4 import S4Block as S4
from models.layers.s4d import S4D
from models.layers.s4d_layer import S4DLayer

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class S4DRecurrentBackbone(nn.Module):

    def __init__(
        self,
        d_input,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
        kernel_type='recurrent',
        training=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        assert kernel_type == "recurrent", "Only recurrent kernel type is supported"
        for _ in range(n_layers):
            self.s4_layers.append(
                S4DLayer(d_model, dropout=dropout, transposed=True, kernel_type="recurrent", training=training, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
    
    def prepare_for_inference(self):
        """Prepare the backbone for inference by setting the S4D layers to inference mode."""
        for layer in self.s4_layers:
            layer.prepare_for_inference()

    def reset_state(self, batch_size, device):
        """
        Reset the state of all S4D layers in the backbone.
        This should be called at the beginning of each new sequence.
        """
        for layer in self.s4_layers:
            layer.reset_state(batch_size, device)

    def forward_step(self, x_t):
        """
        Input x_t is shape (B, d_input) for a single time step.
        Output is shape (B, d_model) for a single time step.
        """

        # Linear encoder
        x_t = self.encoder(x_t)  # (B, d_input) -> (B, d_model)

        # S4D layers
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):

            z = x_t
            if self.prenorm:
                z = norm(z)

            # S4D single step forward
            z = layer.forward_step(z) # (B, d_model)

            # dropout
            if self.training:
                z = dropout(z.unsqueeze(-1)).squeeze(-1)

            # Residual connection
            x_t = z + x_t

            if not self.prenorm:
                x_t = norm(x_t)
                
        return x_t

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        Output is shape (B, L, d_model)
        """
        B, L, D = x.shape

        self.reset_state(B, x.device)

        outputs = torch.empty(B, L, self.d_model, dtype=x.dtype, device=x.device)
        for (t, x_t) in enumerate(x.unbind(dim=1)):
            outputs[:, t, :] = self.forward_step(x_t)  # (B, d_input) -> (B, d_model)

        return outputs  # (B, L, d_model)