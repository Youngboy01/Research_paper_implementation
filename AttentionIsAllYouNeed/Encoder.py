import torch.nn as nn
from LayerNormalisation import LayerNormalisation
from MHSA import MultiHeadAttention
from ResidualConnection import ResidualConnection
from FFN import FFN
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, mha: MultiHeadAttention, ffn: FFN, dropout: float):
        super().__init__()
        self.mha = mha
        self.ffn = ffn
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)

    def forward(self, x, input_mask):
        x = self.residual_connection1(x, lambda x: self.mha(x, x, x, input_mask))
        x = self.residual_connection2(x, self.ffn)
        return x
class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        # this Encoder stacks multiple layers which are created in encoder block, processes input through them, and normalizes the final output.