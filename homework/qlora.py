from pathlib import Path
import torch
import torch.nn as nn
import math

from .bignet import BIGNET_DIM, LayerNorm
from .low_precision import Linear4Bit, block_dequantize_4bit


class QLoRALinear(Linear4Bit):
    def __init__(self, in_features, out_features, lora_dim, group_size=16, bias=True):
        super().__init__(in_features, out_features, bias=bias, group_size=group_size)
        self.requires_grad_(False)

        # LoRA adapters in float32 for gradient accuracy
        self.lora_a = nn.Linear(in_features, lora_dim, bias=False).float()
        self.lora_b = nn.Linear(lora_dim, out_features, bias=False).float()

        # Init
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        # Make trainable
        for p in self.lora_a.parameters():
            p.requires_grad = True
        for p in self.lora_b.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        weight = block_dequantize_4bit(self.weight_q4, self.weight_norm)
        weight = weight.view(self._shape)

        base_out = torch.nn.functional.linear(x, weight, self.bias)
        lora_out = self.lora_b(self.lora_a(x.float()))

        return (base_out + lora_out).to(input_dtype)


class QLoRABigNet(torch.nn.Module):
    def __init__(self, lora_dim=32, group_size=16, bias=True):
        super().__init__()

        def make_block(index):
            use_lora = index in [0, 2, 4]  # LoRA only in these 3 blocks
            if use_lora:
                return nn.Sequential(
                    QLoRALinear(BIGNET_DIM, BIGNET_DIM, lora_dim, group_size, bias),
                    nn.ReLU(),
                    Linear4Bit(BIGNET_DIM, BIGNET_DIM, bias=bias, group_size=group_size),
                    nn.ReLU(),
                    Linear4Bit(BIGNET_DIM, BIGNET_DIM, bias=bias, group_size=group_size),
                )
            else:
                return nn.Sequential(
                    Linear4Bit(BIGNET_DIM, BIGNET_DIM, bias=bias, group_size=group_size),
                    nn.ReLU(),
                    Linear4Bit(BIGNET_DIM, BIGNET_DIM, bias=bias, group_size=group_size),
                    nn.ReLU(),
                    Linear4Bit(BIGNET_DIM, BIGNET_DIM, bias=bias, group_size=group_size),
                )

        # Assemble 6 blocks (3 with LoRA, 3 without)
        layers = []
        for i in range(6):
            layers.append(nn.Sequential(make_block(i), LayerNorm(BIGNET_DIM)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
