import torch
import torch.nn as nn
import math
from pathlib import Path
from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit, block_dequantize_4bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = False,  # Keep this False for memory
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # LoRA adapters (float32 and trainable)
        self.lora_a = nn.Linear(in_features, lora_dim, bias=False).float()
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False).float()

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        for p in self.lora_a.parameters():
            p.requires_grad = True
        for p in self.lora_b.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        weight = block_dequantize_4bit(self.weight_q4, self.weight_norm).view(self._shape)
        base_out = torch.nn.functional.linear(x, weight, self.bias)
        lora_out = self.lora_b(self.lora_a(x.float()))
        return (base_out + lora_out).to(input_dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size, bias=False),
                nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size, bias=False),
                nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size, bias=False),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 2, group_size: int = 256):  # Adjusted
        super().__init__()
        self.model = nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),  # Add 1st norm
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),  # Add 2nd norm
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
