import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
      #expects a token of indices
        return self.embedding(x) * math.sqrt(self.d_model)#maps each token index to learned vector of size d_model. Each token has its vector which is learned during training
    #multiplied by sqrt(d_model) coz it keeps variance stable as the depth grows