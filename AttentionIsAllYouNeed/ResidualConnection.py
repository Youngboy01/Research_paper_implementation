import torch.nn as nn
from LayerNormalisation import LayerNormalisation
class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalisation = LayerNormalisation(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.normalisation(x)))
