from __future__ import annotations

from typing import Dict

import torch


def init_metric_state(task_type: str, device: torch.device) -> Dict[str, torch.Tensor]:
    state = {
        "loss_sum": torch.zeros(1, device=device),
        "num_items": torch.zeros(1, device=device),
    }
    if task_type == "multiclass":
        state["correct"] = torch.zeros(1, device=device)
    else:
        state["tp"] = torch.zeros(1, device=device)
        state["fp"] = torch.zeros(1, device=device)
        state["fn"] = torch.zeros(1, device=device)
        state["exact_match"] = torch.zeros(1, device=device)
    return state


def update_metric_state(
    state: Dict[str, torch.Tensor],
    task_type: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    threshold: float,
) -> None:
    batch_size = targets.shape[0]
    state["loss_sum"] += loss.detach() * batch_size
    state["num_items"] += batch_size

    if task_type == "multiclass":
        preds = logits.argmax(dim=-1)
        state["correct"] += (preds == targets).sum()
        return

    preds = (torch.sigmoid(logits) >= threshold).float()
    state["tp"] += ((preds == 1) & (targets == 1)).sum()
    state["fp"] += ((preds == 1) & (targets == 0)).sum()
    state["fn"] += ((preds == 0) & (targets == 1)).sum()
    state["exact_match"] += (preds == targets).all(dim=-1).sum()


def compute_epoch_metrics(task_type: str, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    num_items = state["num_items"].clamp_min(1.0)
    metrics = {"loss": state["loss_sum"] / num_items}
    if task_type == "multiclass":
        metrics["accuracy"] = state["correct"] / num_items
        return metrics

    tp = state["tp"]
    fp = state["fp"]
    fn = state["fn"]
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = (2 * precision * recall) / (precision + recall).clamp_min(1e-6)
    metrics["precision_micro"] = precision
    metrics["recall_micro"] = recall
    metrics["f1_micro"] = f1
    metrics["exact_match"] = state["exact_match"] / num_items
    return metrics
