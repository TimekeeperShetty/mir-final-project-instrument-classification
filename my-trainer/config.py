import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class TrainConfig:
    seed: int = 0
    precision: str = "bf16-mixed"
    max_epochs: int = 1
    log_every_n_steps: int = 50
    grad_clip: float = 0.0
    accumulate_grad_batches: int = 1

    # data
    batch_size: int = 32
    num_workers: int = 8

    # optional: val / demo
    val_every_n_steps: int = 0   # 0 disables validation
    demo_every_n_steps: int = 0  # 0 disables demo callback

    # optimization
    lr: float = 3e-4

    # system
    save_dir: Optional[str] = None
    ckpt_every_n_steps: int = 2000


def parse_train_config(cfg: Dict[str, Any]) -> TrainConfig:
    # minimal + strict-ish
    tc = TrainConfig()
    for k, v in cfg.items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    return tc