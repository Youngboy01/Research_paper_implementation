import torch.nn as nn
from InputEmbeddings import InputEmbeddings
from PositionalEncodings import Positional_Encodings
from Encoder import Encoder,EncoderLayer
from Decoder import Decoder,DecoderLayer
from LinearLayer import LinearLayer
from MHSA import MultiHeadAttention
from Transformer import Transformer
from FFN import FFN
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, heads: int = 8, dropout: float = 0.1, d_ffn: int = 2048) -> Transformer:
     #Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, tgt_vocab_size)
    #create positional encoding layers
    src_position = Positional_Encodings(src_seq_len, d_model, dropout)
    target_position = Positional_Encodings(tgt_seq_len, d_model, dropout)
    encoder_blocks = []
    for _ in range(N):
        encoder_blocks.append(EncoderLayer(d_model, MultiHeadAttention(d_model, heads, dropout), FFN(d_model, d_ffn, dropout), dropout))
    decoder_blocks = []
    for _ in range(N):
        decoder_blocks.append(DecoderLayer(d_model, MultiHeadAttention(d_model, heads, dropout), MultiHeadAttention(d_model, heads, dropout), FFN(d_model, d_ffn, dropout), dropout))
    #create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    # create the projection layer
    linear = LinearLayer(d_model, tgt_vocab_size)
    #create the transformer
    transformer = Transformer(src_embed, src_position, encoder, decoder, target_embed, target_position, linear)
    # randomly initialise the parameters using xavier initialisation
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
