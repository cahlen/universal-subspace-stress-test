"""
Phase 2 (variant): "Order 1-2 HOSVD" as the paper describes.

Given a 3-way tensor X ∈ R^{N × d_out × d_in} formed by stacking N weight
matrices, HOSVD of "Order 1-2" (paper's default) means we unfold along modes
1 and 2 (the d_out and d_in axes), do thin SVD of each unfolding, and use
those factor matrices.

This is a *different* claim than what we tested in run.py (mode-0 unfolding
finds the shared subspace across models). The question here: do the d_out and
d_in factors of random-init ViT stacks show any structure that an iid Gaussian
tensor does not?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hosvd import thin_svd

N_MODELS = 100
LAYERS_TO_PROBE = [
    ("block00_q_proj",  "encoder.layer.0.attention.attention.query.weight"),
    ("block05_q_proj",  "encoder.layer.5.attention.attention.query.weight"),
    ("block11_q_proj",  "encoder.layer.11.attention.attention.query.weight"),
]
OUTDIR = Path(__file__).parent / "results"
DTYPE = torch.float64


def get_attr(model, path):
    obj = model
    for p in path.split("."):
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    return obj


def unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Matricize a tensor along the given mode (0-indexed)."""
    return torch.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def main():
    from transformers import ViTConfig, ViTModel

    config = ViTConfig()
    layer_stacks = {n: [] for n, _ in LAYERS_TO_PROBE}

    t0 = time.time()
    for seed in range(N_MODELS):
        torch.manual_seed(seed)
        model = ViTModel(config, add_pooling_layer=False)
        for name, path in LAYERS_TO_PROBE:
            W = get_attr(model, path).detach().to(dtype=DTYPE).clone()
            layer_stacks[name].append(W)
        del model
        if (seed + 1) % 20 == 0:
            print(f"  [{seed+1}/{N_MODELS}] {time.time()-t0:.1f}s")

    report = {}
    for name, mats in layer_stacks.items():
        X = torch.stack(mats, dim=0)           # (N, d_out, d_in)
        Xc = X - X.mean(dim=0, keepdim=True)

        null_X = torch.randn(*X.shape, dtype=DTYPE, generator=torch.Generator().manual_seed(99)) * X.std()
        null_Xc = null_X - null_X.mean(dim=0, keepdim=True)

        report[name] = {"shape": list(X.shape), "modes": {}}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for mi, mode in enumerate([1, 2]):
            U_vit, S_vit, _ = thin_svd(unfold(Xc, mode))
            U_null, S_null, _ = thin_svd(unfold(null_Xc, mode))

            evr_vit = (S_vit**2) / (S_vit**2).sum()
            evr_null = (S_null**2) / (S_null**2).sum()

            k = min(len(S_vit), 60)
            ax_sv = axes[mi, 0]
            ax_cum = axes[mi, 1]

            ax_sv.plot(np.arange(1, k+1), S_vit[:k].cpu().numpy(), "o-", label="ViT")
            ax_sv.plot(np.arange(1, k+1), S_null[:k].cpu().numpy(), "s--", label="iid null")
            ax_sv.set_title(f"{name}, mode-{mode} unfolding — singular values")
            ax_sv.set_xlabel("component"); ax_sv.set_ylabel("σ"); ax_sv.grid(alpha=.3); ax_sv.legend()

            ax_cum.plot(np.arange(1, k+1), np.cumsum(evr_vit[:k].cpu().numpy()), "o-", label="ViT")
            ax_cum.plot(np.arange(1, k+1), np.cumsum(evr_null[:k].cpu().numpy()), "s--", label="iid null")
            ax_cum.set_title(f"mode-{mode} — cumulative EVR")
            ax_cum.set_xlabel("component"); ax_cum.set_ylim(0, 1.01); ax_cum.grid(alpha=.3); ax_cum.legend()

            report[name]["modes"][f"mode{mode}"] = {
                "vit_top16_evr": float(evr_vit[:16].sum().item()),
                "null_top16_evr": float(evr_null[:16].sum().item()),
                "vit_top1_evr":  float(evr_vit[0].item()),
                "null_top1_evr": float(evr_null[0].item()),
            }
            print(f"  {name} mode-{mode}: ViT top1={evr_vit[0]:.4f}  null top1={evr_null[0]:.4f}   "
                  f"ViT top16={evr_vit[:16].sum():.4f}  null top16={evr_null[:16].sum():.4f}")

        fig.tight_layout()
        fig.savefig(OUTDIR / f"order12_{name}.png", dpi=130)
        plt.close(fig)

    (OUTDIR / "phase2_order12_report.json").write_text(json.dumps(report, indent=2))
    print("[done]")


if __name__ == "__main__":
    main()
