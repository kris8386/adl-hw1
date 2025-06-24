from pathlib import Path
import torch
from .qlora import QLoRABigNet  

def load(path: Path | None):
    # Extra credit: A BigNet variant that uses <4 bits/parameter and retains decent accuracy
    model = QLoRABigNet(lora_dim=32, group_size=16)
    if path is not None:
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    return model

