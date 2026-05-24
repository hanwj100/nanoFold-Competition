from __future__ import annotations

from typing import Any, Dict

import torch

from nanofold.model import NanoFoldBaseline, distogram_loss
from nanofold.residue_constants import ATOM14_NUM_SLOTS, CA_ATOM14_SLOT


def _atom14_from_ca(pred_ca: torch.Tensor) -> torch.Tensor:
    pred_atom14 = pred_ca.unsqueeze(2).expand(-1, -1, ATOM14_NUM_SLOTS, -1).contiguous()
    pred_atom14[:, :, CA_ATOM14_SLOT, :] = pred_ca
    return pred_atom14


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    m = cfg["model"]
    return NanoFoldBaseline(
        d_model=int(m["d_model"]),
        n_layers=int(m["n_layers"]),
        n_heads=int(m["n_heads"]),
        dropout=float(m.get("dropout", 0.1)),
    )


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    o = cfg["optim"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(o["lr"]),
        weight_decay=float(o.get("weight_decay", 1e-2)),
    )


def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    pred_ca = model(batch["aatype"], batch["msa"], batch["deletions"], batch["residue_mask"])
    pred_atom14 = _atom14_from_ca(pred_ca)

    if not training:
        return {"pred_atom14": pred_atom14}

    loss = distogram_loss(
        pred_ca=pred_ca,
        true_ca=batch["ca_coords"],
        ca_mask=batch["ca_mask"],
        residue_mask=batch["residue_mask"],
    )
    return {"pred_atom14": pred_atom14, "loss": loss}
