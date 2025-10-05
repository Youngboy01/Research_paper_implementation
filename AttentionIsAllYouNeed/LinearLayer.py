import torch.nn as nn
import torch
#output of decoder is (sequence,d_model) but we want to map words back to vocabulary. This linear layer converts embeddingn into position of vocab.
#The Linear layer maps these hidden features to vocabulary logits, enabling the model to learn meaningful associations between internal features and output tokens.
class LinearLayer(nn.Module):
  def __init__(self,d_model,vocab_size):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.linear = nn.Linear(d_model,vocab_size)
  def forward(self,x):
    #convert (batch,seq_len,d_model)->(batch,seq_len,vocab_size)
    return torch.log_softmax(self.linear(x),dim=-1)