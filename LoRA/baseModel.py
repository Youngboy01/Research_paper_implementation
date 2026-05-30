import torch.nn as nn
from LoraModule import LoRALinear


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class LoRAMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank=2, alpha=1.0):
        super(LoRAMLP, self).__init__()
        self.fc1 = LoRALinear(nn.Linear(input_dim, hidden_dim), rank, alpha)
        self.relu = nn.ReLU()
        self.fc2 = LoRALinear(nn.Linear(hidden_dim, output_dim), rank, alpha)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
