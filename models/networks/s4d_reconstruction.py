import torch
import torch.nn as nn

from models.layers.s4 import S4Block as S4
from models.layers.s4d import S4D
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
        lr=0.001
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
        )

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_input)

    def forward(self, x):
        # (B,L,D)
        x = self.backbone(x)

        # reconstruct
        x = self.decoder(x)

        return x