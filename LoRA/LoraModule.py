import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=2, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        self.scale = alpha / rank

    def forward(self, x):
        return self.scale * (x @ self.A @ self.B)


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=2, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRA(
            linear_layer.in_features, linear_layer.out_features, rank, alpha
        )
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)
