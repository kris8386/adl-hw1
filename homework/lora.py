import torch
import math
from pathlib import Path

from .bignet import BIGNET_DIM, LayerNorm
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(self, in_features, out_features, lora_dim, bias=True):
        super().__init__(in_features, out_features, bias)

        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)

        self.lora_a = self.lora_a.float()
        self.lora_b = self.lora_b.float()

        for param in self.parameters():
            param.requires_grad = False
        for param in list(self.lora_a.parameters()) + list(self.lora_b.parameters()):
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        base_out = super().forward(x)
        lora_out = self.lora_b(self.lora_a(x.float()))
        return (base_out + lora_out).to(input_dtype)


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self, lora_dim=32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),  # Block 1
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),  # Block 2
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),  # Block 3
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),  # Block 4
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),  # Block 5
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),  # Block 6
        )

    def forward(self, x):
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net