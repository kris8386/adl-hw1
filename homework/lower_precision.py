from pathlib import Path
import torch
from .qlora import QLoRABigNet  # Adjust if QLoRABigNet is in a different location


def load(path: Path | None):
    # Extra credit: A BigNet variant that uses <4 bits/parameter and retains decent accuracy
    model = QLoRABigNet(lora_dim=2, group_size=8192)
    if path is not None:
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    return model

