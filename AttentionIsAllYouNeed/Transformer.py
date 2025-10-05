import torch.nn as nn
from InputEmbeddings import InputEmbeddings
from PositionalEncodings import Positional_Encodings
from Encoder import Encoder
from Decoder import Decoder
from LinearLayer import LinearLayer
class Transformer(nn.Module):
    def __init__(self, src_embed: InputEmbeddings, src_position: Positional_Encodings, encoder: Encoder, decoder: Decoder, target_embed: InputEmbeddings, target_position: Positional_Encodings, linear: LinearLayer):
        super().__init__()
        self.src_embed = src_embed
        self.src_position = src_position
        self.encoder = encoder
        self.decoder = decoder
        self.target_embed = target_embed
        self.target_position = target_position
        self.linear = linear

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target_mask, target):
        target = self.target_embed(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def linearlayer(self, x):
        return self.linear(x)