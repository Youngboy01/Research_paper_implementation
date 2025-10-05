import torch.nn as nn
from LayerNormalisation import LayerNormalisation
from MHSA import MultiHeadAttention
from ResidualConnection import ResidualConnection
from FFN import FFN
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, ffn: FFN, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ffn = ffn
        self.residual_connection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, input_mask, output_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, output_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, input_mask))
        x = self.residual_connection[2](x, self.ffn)
        return x
class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation(d_model)

    def forward(self, x, encoder_output, input_mask, output_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, input_mask, output_mask)
        return self.norm(x)
