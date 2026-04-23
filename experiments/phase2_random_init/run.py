"""
Phase 2: Random-Init ViT Null Test.

Question: The paper claims 500 randomly-initialized (untrained) ViTs share a
low-rank subspace. Is this non-trivial, or does it trivially follow from the
fact that all models share an init distribution?

Protocol:
  1. Instantiate N freshly-seeded ViT-B/16 models (untrained).
  2. From each, extract weight matrices of several specific layers.
  3. For each layer, stack matrices, zero-center, SVD along model axis.
  4. Record singular-value spectrum / explained-variance ratio.
  5. Generate a matched-covariance iid-Gaussian null (Marchenko-Pastur regime)
     and compare spectra.

If the null reproduces the ViT spectrum → the "random-init universal subspace"
claim is a trivial consequence of init statistics, not a learned property.

Output: JSON with raw singular values, PNG scree-curve plots.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hosvd import explained_variance_ratio, gaussian_null_singular_values, model_axis_svd, thin_svd

# ---- Config ----
N_MODELS = 100                       # start at 100; scale up if fast enough
LAYERS_TO_PROBE = [
    # (human-name, attribute-path)
    ("block00_q_proj",    "encoder.layer.0.attention.attention.query.weight"),
    ("block05_q_proj",    "encoder.layer.5.attention.attention.query.weight"),
    ("block11_q_proj",    "encoder.layer.11.attention.attention.query.weight"),
    ("block05_mlp_fc1",   "encoder.layer.5.intermediate.dense.weight"),
    ("block05_mlp_fc2",   "encoder.layer.5.output.dense.weight"),
]
OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(exist_ok=True)
DEVICE = "cpu"                       # SVD on CPU; float64 for numerical stability
DTYPE = torch.float64


def get_attr(model, path: str):
    obj = model
    for p in path.split("."):
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    return obj


def main():
    from transformers import ViTConfig, ViTModel

    config = ViTConfig()  # defaults = ViT-B/16
    print(f"[cfg] hidden={config.hidden_size}  layers={config.num_hidden_layers}  "
          f"heads={config.num_attention_heads}  seed-count={N_MODELS}")

    layer_stacks: dict[str, list[torch.Tensor]] = {name: [] for name, _ in LAYERS_TO_PROBE}

    t0 = time.time()
    for seed in range(N_MODELS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ViTModel(config, add_pooling_layer=False)
        for name, path in LAYERS_TO_PROBE:
            W = get_attr(model, path).detach().to(dtype=DTYPE, device=DEVICE).clone()
            layer_stacks[name].append(W)
        del model
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  [{seed + 1:4d}/{N_MODELS}]  elapsed={time.time()-t0:.1f}s")

    print(f"[done-init] {N_MODELS} ViTs instantiated in {time.time()-t0:.1f}s")

    report: dict = {"n_models": N_MODELS, "layers": {}}

    for name, mats in layer_stacks.items():
        stacked = torch.stack(mats, dim=0)                      # (N, d_out, d_in)
        print(f"\n[layer {name}]  stacked shape = {tuple(stacked.shape)}")

        # --- ViT spectrum ---
        mean, U, S, Vh = model_axis_svd(stacked, zero_center=True)
        evr = explained_variance_ratio(S)
        S_np = S.cpu().numpy()
        evr_np = evr.cpu().numpy()

        # --- mean contribution ---
        mean_norm = torch.linalg.norm(mean).item()
        flat_norm = torch.linalg.norm(stacked.reshape(N_MODELS, -1)).item()
        mean_frac = (mean_norm * np.sqrt(N_MODELS)) / flat_norm   # energy fraction of the mean

        # --- iid-Gaussian null with matched std of ENTRIES ---
        sigma = stacked.reshape(-1).std().item()
        D = stacked.shape[1] * stacked.shape[2]
        null_S = gaussian_null_singular_values(N_MODELS, D, sigma=sigma, seed=12345).cpu().numpy()
        null_evr = (null_S**2) / (null_S**2).sum()

        report["layers"][name] = {
            "shape": list(stacked.shape),
            "entry_std": float(sigma),
            "mean_energy_fraction": float(mean_frac),
            "vit_singular_values": S_np.tolist(),
            "vit_evr": evr_np.tolist(),
            "null_singular_values": null_S.tolist(),
            "null_evr": null_evr.tolist(),
            "vit_top1_evr": float(evr_np[0]),
            "null_top1_evr": float(null_evr[0]),
            "vit_top16_evr": float(evr_np[:16].sum()),
            "null_top16_evr": float(null_evr[:16].sum()),
        }

        print(f"  entry std          = {sigma:.4e}")
        print(f"  mean fraction      = {mean_frac:.4f}")
        print(f"  ViT   top-1 EVR    = {evr_np[0]:.4f}      null top-1 EVR = {null_evr[0]:.4f}")
        print(f"  ViT   top-16 EVR   = {evr_np[:16].sum():.4f}     null top-16 EVR = {null_evr[:16].sum():.4f}")
        print(f"  ViT   max σ / min σ = {S_np[0] / S_np[-1]:.3f}   "
              f"null max σ / min σ = {null_S[0] / null_S[-1]:.3f}")

        # --- Plot ---
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        k = min(len(S_np), 60)
        ax[0].plot(np.arange(1, k + 1), S_np[:k], "o-", label="ViT (random init)")
        ax[0].plot(np.arange(1, k + 1), null_S[:k], "s--", label=f"iid-Gaussian null (σ={sigma:.3e})")
        ax[0].set_xlabel("Component")
        ax[0].set_ylabel("Singular value")
        ax[0].set_title(f"{name} — top-{k} singular values")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        ax[1].plot(np.arange(1, k + 1), np.cumsum(evr_np[:k]), "o-", label="ViT (random init)")
        ax[1].plot(np.arange(1, k + 1), np.cumsum(null_evr[:k]), "s--", label="iid-Gaussian null")
        ax[1].set_xlabel("Component")
        ax[1].set_ylabel("Cumulative EVR")
        ax[1].set_title(f"{name} — cumulative explained variance")
        ax[1].set_ylim(0, 1.01)
        ax[1].legend()
        ax[1].grid(alpha=0.3)

        fig.tight_layout()
        out_png = OUTDIR / f"scree_{name}.png"
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"  plot saved → {out_png}")

    out_json = OUTDIR / "phase2_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\n[done] report → {out_json}")


if __name__ == "__main__":
    main()
