import torch
import torch.nn as nn
import math
class Positional_Encodings(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len #max length of input sequences
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)#randomly zero out some featires during training for better generalisation and reduce overfitting
        pe = torch.zeros(seq_len, d_model)# create a matrix of length seq_len,d_model
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)#vector of shape(seq_len,1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#vector of shape(d_model/2) #calculated in log space for numerical satbility
        pe[:, 0::2] = torch.sin(pos * denominator)#sin to even positions
        pe[:, 1::2] = torch.cos(pos * denominator)#cos to odd positions

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe:torch.Tensor
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)