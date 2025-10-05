import torch.nn as nn
import torch
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_k = d_model // heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def Attention(query, key, value, mask=None, dropout=None):
        #query: (batch_size, heads, seq_len, d_k)
        #key: (batch_size, heads, seq_len, d_k)
        #value: (batch_size, heads, seq_len, d_k)
        #(batch_size, heads, seq_len, d_k) * (batch_size, heads, d_k, seq_len) -> (batch_size, heads, seq_len, seq_len)
        d_k = query.shape[-1]
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = torch.softmax(score, dim=-1)#(batch,heads,seq_len,seq_len)
        if dropout is not None:
            score = dropout(score)
        return torch.matmul(score, value), score

    def forward(self, q, k, v, mask=None):#mask for if we dont want some words to interact with other words
        query = self.W_q(q)#(batch_size, seq_len, d_model)->(batch_size, seq_len,d_model)
        key = self.W_k(k)#(batch_size, seq_len, d_model)->(batch_size, seq_len,d_model)
        value = self.W_v(v)#(batch_size, seq_len, d_model)->(batch_size, seq_len,d_model)
        #we wnat to split the embeddings into h parts not the sentence

        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)#(batch_size, seq_len, d_model)->(batch_size, seq_len, heads, d_k)->(batch_size, heads, seq_len, d_k)
        #each head will see the full sentence . So each word in the sentence but only a smaller part of the embedding
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)#(batch_size, seq_len, d_model)->(batch_size, seq_len, heads, d_k)->(batch_size, heads, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)#(batch_size, seq_len, d_model)->(batch_size, seq_len, heads, d_k)->(batch_size, heads, seq_len, d_k)

        x, self.attention_score = MultiHeadAttention.Attention(query, key, value, mask, self.dropout)
        #now we covert (batch_size, heads, seq_len, d_k)->(batch_size,seq_len,heads,d_k)->(batch_size,seq_len,d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)
        #-1 allows dynamic, flexible reshaping without explicitly specifying that dimension.

        return self.W_o(x)