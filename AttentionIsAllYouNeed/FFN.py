import torch.nn as nn
import torch
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)#W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)#W2 and B2

    def forward(self, x):
        #(batch,seq,d_model) -> (batch,seq,d_ffn)-> (batch,seq,d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))