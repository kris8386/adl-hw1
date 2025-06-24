import torch
import torch.nn as nn
import math
from .low_precision import Linear4Bit, block_dequantize_4bit
from .bignet import BIGNET_DIM, LayerNorm


class QLoRALinear(Linear4Bit):
    def __init__(self, in_features, out_features, lora_dim, group_size=128, bias=False):
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # LoRA adapters (float32)
        self.lora_a = nn.Linear(in_features, lora_dim, bias=False).float()
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False).float()

        # Init
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        # Make LoRA trainable
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


class QLoRABigNet(nn.Module):
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

    def __init__(self, lora_dim=4, group_size=128):
        super().__init__()
        self.model = nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    model = QLoRABigNet(lora_dim=4, group_size=128)
    if path is not None:
        model.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return model
from pathlib import Path
