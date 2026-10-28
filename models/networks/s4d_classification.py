import torch
import torch.nn as nn

from models.backbones.s4d_backbone import S4DBackbone


class Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.001,
        kernel_type='conv',
    ):
        super().__init__()

        self.prenorm = prenorm

        # Backbone
        self.backbone = S4DBackbone(
            d_input=d_input,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            lr=lr,
            kernel_type=kernel_type,
        )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def prepare_for_inference(self):
        """Prepare the model for inference by caching the discrete parameters of each S4D layer."""
        self.backbone.prepare_for_inference()

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            self.prepare_for_inference()

        return self

    def forward(self, x):
        # (B,L,D)
        x = self.backbone(x)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x