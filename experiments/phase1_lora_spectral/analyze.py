"""
Phase 1: Spectral analysis of N Mistral-7B LoRAs from Lots-of-LoRAs.

Fast path: for an (N, d_out, d_in) tensor with N << D = d_out*d_in, the N
singular values are recovered from the N×N Gram matrix eigendecomposition.
That lets us avoid materializing the (N, D) SVD factors. Runs on GPU.

Reconstruction error at rank k is computable from the singular values alone:

    ||X - X_k||_F^2 = Σ_{i > k} σ_i^2    (for centered X)
    relative error  = sqrt(sum_{i>k} σ_i^2 / (N·||mean||^2 + Σ σ_i^2))

where the denominator uses the original (uncentered) Frobenius norm.

Null model: rank-r Gaussian LoRA updates ΔW = B @ A with A, B iid-Gaussian,
matching the empirical entrywise variances of the real LoRAs' A and B.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LORA_DIR = Path(os.environ.get("LORA_DIR", "./data/loras"))
OUTDIR   = Path(__file__).parent / "results"
OUTDIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32   # float64 is overkill; float32 on GPU is fast and accurate enough


def parse_lora(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    sd = load_file(path)
    out: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, val in sd.items():
        m = re.match(r"^base_model\.model\.(.*)\.(lora_[AB])\.weight$", key)
        if m is None:
            continue
        mod_name, ab = m.group(1), m.group(2)
        out[mod_name][ab] = val.to(dtype=DTYPE)
    return {k: (v["lora_A"], v["lora_B"]) for k, v in out.items() if "lora_A" in v and "lora_B" in v}


def spectrum_via_gram(X_flat: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    Given X_flat (N, D) and its column mean, return singular values of (X_flat - mean).
    Computes the N×N Gram matrix and eigendecomposes it. Memory: O(N² + ND).
    """
    Xc = X_flat - mean                                  # still (N, D)
    gram = Xc @ Xc.T                                    # (N, N)
    eigvals = torch.linalg.eigvalsh(gram)               # ascending
    eigvals = torch.clamp(eigvals.flip(0), min=0.0)
    return torch.sqrt(eigvals)                          # (N,)


def analyze(
    max_loras: int = 100,
    detail_layers: list[str] | None = None,
    null_seed_base: int = 42,
):
    lora_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])
    lora_dirs = lora_dirs[:max_loras]
    print(f"[load] parsing {len(lora_dirs)} LoRAs from {LORA_DIR} device={DEVICE}", flush=True)

    modules_A: dict[str, list[torch.Tensor]] = defaultdict(list)
    modules_B: dict[str, list[torch.Tensor]] = defaultdict(list)

    t0 = time.time()
    for i, d in enumerate(lora_dirs):
        parsed = parse_lora(d / "adapter_model.safetensors")
        for mod, (A, B) in parsed.items():
            modules_A[mod].append(A)
            modules_B[mod].append(B)
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{len(lora_dirs)}  ({time.time()-t0:.1f}s)", flush=True)

    complete_modules = sorted([m for m in modules_A if len(modules_A[m]) == len(lora_dirs)])
    print(f"[load-done] {len(complete_modules)} complete modules", flush=True)

    if detail_layers is None:
        detail_layers = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.15.self_attn.q_proj",
            "model.layers.15.self_attn.v_proj",
            "model.layers.31.self_attn.q_proj",
            "model.layers.31.self_attn.v_proj",
            "model.layers.15.mlp.gate_proj",
            "model.layers.15.mlp.down_proj",
        ]

    report: dict = {"n_loras": len(lora_dirs), "layers": {}}
    all_rows = []
    analyze_t0 = time.time()

    for idx, mod in enumerate(complete_modules):
        A_list = modules_A[mod]
        B_list = modules_B[mod]

        A_stack = torch.stack(A_list, dim=0).to(DEVICE)          # (N, rank, d_in)
        B_stack = torch.stack(B_list, dim=0).to(DEVICE)          # (N, d_out, rank)
        rank = A_stack.shape[1]
        N, d_out = B_stack.shape[0], B_stack.shape[1]
        d_in = A_stack.shape[2]

        dW = torch.einsum("nor,nri->noi", B_stack, A_stack)      # (N, d_out, d_in)
        Xf = dW.reshape(N, -1)
        mean = Xf.mean(dim=0)
        S = spectrum_via_gram(Xf, mean)                          # (N,)
        uncentered_norm_sq = (Xf.pow(2).sum()).item()
        mean_energy = (N * mean.pow(2).sum()).item()
        total_sq = (S.pow(2).sum()).item()                       # = ||Xc||_F^2
        evr = (S.pow(2) / S.pow(2).sum()).cpu().numpy()

        # ---- Null: rank-r ΔW with iid Gaussian A, B (matched variance to empirical) ----
        sigma_A = float(A_stack.reshape(-1).std().item())
        sigma_B = float(B_stack.reshape(-1).std().item())
        seed = null_seed_base + idx
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        A_null = torch.randn(N, rank, d_in, device=DEVICE, dtype=DTYPE, generator=g) * sigma_A
        B_null = torch.randn(N, d_out, rank, device=DEVICE, dtype=DTYPE, generator=g) * sigma_B
        dW_null = torch.einsum("nor,nri->noi", B_null, A_null)
        Xn = dW_null.reshape(N, -1)
        mean_n = Xn.mean(dim=0)
        S_null = spectrum_via_gram(Xn, mean_n)
        evr_null = (S_null.pow(2) / S_null.pow(2).sum()).cpu().numpy()

        # ---- Reconstruction errors at various k (computed from singular values) ----
        # For centered X, ||X - X_k||_F^2 = sum_{i>k} σ_i^2
        # Full model reconstruction error vs original uncentered = sqrt(sum_{i>k} σ_i^2 / uncentered_norm_sq)
        # if we also reconstruct the mean (i.e., add it back).
        S_sq = S.cpu().numpy() ** 2
        total_with_mean = uncentered_norm_sq  # because ||X||² = N||μ||² + ||Xc||² = N||μ||² + Σσ²
        recon_errs = {}
        for k in [1, 2, 4, 8, 16, 32, 64, min(N - 1, 96)]:
            if k <= N:
                residual_sq = S_sq[k:].sum()
                recon_errs[int(k)] = float(np.sqrt(residual_sq / total_with_mean))

        row = {
            "layer": mod,
            "N": N, "d_out": d_out, "d_in": d_in, "rank": rank,
            "uncentered_norm_sq": uncentered_norm_sq,
            "mean_energy_fraction": mean_energy / uncentered_norm_sq,
            "top1_vit":  float(evr[0]),  "top1_null":  float(evr_null[0]),
            "top4_vit":  float(evr[:4].sum()),  "top4_null":  float(evr_null[:4].sum()),
            "top16_vit": float(evr[:16].sum()), "top16_null": float(evr_null[:16].sum()),
            "top32_vit": float(evr[:32].sum()), "top32_null": float(evr_null[:32].sum()),
            "top64_vit": float(evr[:64].sum()) if N >= 64 else None,
            "top64_null": float(evr_null[:64].sum()) if N >= 64 else None,
            "recon_err": recon_errs,
        }
        all_rows.append(row)
        report["layers"][mod] = row

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(complete_modules)}] last={mod}  top16 LoRA={row['top16_vit']:.3f} "
                  f"null={row['top16_null']:.3f}  elapsed={time.time()-analyze_t0:.1f}s", flush=True)

        if mod in detail_layers:
            print(f"\n[detail] {mod}  shape=(N={N}, d_out={d_out}, d_in={d_in})  rank={rank}")
            print(f"  mean energy fraction = {row['mean_energy_fraction']:.4f}")
            print(f"  top-1   EVR  LoRA={row['top1_vit']:.4f}   null={row['top1_null']:.4f}")
            print(f"  top-16  EVR  LoRA={row['top16_vit']:.4f}   null={row['top16_null']:.4f}   ratio={row['top16_vit']/max(row['top16_null'],1e-9):.2f}")
            print(f"  top-32  EVR  LoRA={row['top32_vit']:.4f}   null={row['top32_null']:.4f}   ratio={row['top32_vit']/max(row['top32_null'],1e-9):.2f}")
            print(f"  recon err (rel): " + "  ".join(f"k={k}:{e:.3f}" for k, e in recon_errs.items()))

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            k_show = min(N, 80)
            xs = np.arange(1, k_show + 1)
            ax[0].semilogy(xs, S.cpu().numpy()[:k_show], "o-", label="trained LoRAs")
            ax[0].semilogy(xs, S_null.cpu().numpy()[:k_show], "s--", label=f"rank-r null (σA={sigma_A:.2e}, σB={sigma_B:.2e})")
            ax[0].set_xlabel("component"); ax[0].set_ylabel("σ (log scale)")
            ax[0].set_title(f"{mod}")
            ax[0].legend(); ax[0].grid(alpha=.3)

            ax[1].plot(xs, np.cumsum(evr[:k_show]), "o-", label="trained LoRAs")
            ax[1].plot(xs, np.cumsum(evr_null[:k_show]), "s--", label="rank-r null")
            ax[1].axvline(16, color="gray", ls=":", label="paper rank=16")
            ax[1].set_xlabel("component"); ax[1].set_ylabel("cumulative EVR")
            ax[1].set_ylim(0, 1.01)
            ax[1].set_title(f"{mod}  cumulative EVR")
            ax[1].legend(); ax[1].grid(alpha=.3)
            fig.tight_layout()
            safe = mod.replace(".", "_")
            fig.savefig(OUTDIR / f"scree_{safe}.png", dpi=130)
            plt.close(fig)

        # Free GPU memory before next layer
        del A_stack, B_stack, dW, Xf, mean, S, dW_null, Xn, mean_n, S_null, A_null, B_null
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    (OUTDIR / "phase1_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] report → {OUTDIR/'phase1_report.json'}   elapsed={time.time()-analyze_t0:.1f}s")

    vit_t16  = np.array([r["top16_vit"]  for r in all_rows])
    null_t16 = np.array([r["top16_null"] for r in all_rows])
    vit_t32  = np.array([r["top32_vit"]  for r in all_rows])
    null_t32 = np.array([r["top32_null"] for r in all_rows])
    mean_frac = np.array([r["mean_energy_fraction"] for r in all_rows])

    # Distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(vit_t16, bins=30, alpha=.6, label="trained LoRAs")
    axes[0].hist(null_t16, bins=30, alpha=.6, label="rank-r null")
    axes[0].set_xlabel("top-16 cumulative EVR"); axes[0].set_ylabel("# layers")
    axes[0].set_title(f"top-16 EVR across {len(all_rows)} layers (N={len(lora_dirs)})")
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].hist(vit_t32, bins=30, alpha=.6, label="trained LoRAs")
    axes[1].hist(null_t32, bins=30, alpha=.6, label="rank-r null")
    axes[1].set_xlabel("top-32 cumulative EVR"); axes[1].set_ylabel("# layers")
    axes[1].set_title("top-32 EVR distribution")
    axes[1].legend(); axes[1].grid(alpha=.3)

    axes[2].hist(mean_frac, bins=30)
    axes[2].set_xlabel("mean-energy fraction of ||X||²")
    axes[2].set_title("How much of the norm is the mean LoRA?")
    axes[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "summary_distributions.png", dpi=130)
    plt.close(fig)

    print(f"\n=== SUMMARY across {len(all_rows)} layers ===")
    print(f"  mean energy fraction:     median={np.median(mean_frac):.4f}   mean={mean_frac.mean():.4f}")
    print(f"  top-1  EVR  LoRA median={np.median([r['top1_vit'] for r in all_rows]):.4f}   null median={np.median([r['top1_null'] for r in all_rows]):.4f}")
    print(f"  top-16 EVR  LoRA median={np.median(vit_t16):.4f}   null median={np.median(null_t16):.4f}   ratio={np.median(vit_t16)/np.median(null_t16):.2f}")
    print(f"  top-32 EVR  LoRA median={np.median(vit_t32):.4f}   null median={np.median(null_t32):.4f}   ratio={np.median(vit_t32)/np.median(null_t32):.2f}")
    # Layers with strongest low-rank signal
    top16_above_null = vit_t16 - null_t16
    order = np.argsort(top16_above_null)[::-1]
    print(f"\nLayers where trained LoRAs MOST exceed null (top-16 EVR excess):")
    for i in order[:5]:
        r = all_rows[i]
        print(f"   {r['layer']:60s}  LoRA={r['top16_vit']:.3f}  null={r['top16_null']:.3f}  excess={top16_above_null[i]:+.3f}")
    print(f"Layers where LoRAs LEAST exceed null:")
    for i in order[-5:]:
        r = all_rows[i]
        print(f"   {r['layer']:60s}  LoRA={r['top16_vit']:.3f}  null={r['top16_null']:.3f}  excess={top16_above_null[i]:+.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()
    analyze(max_loras=args.max)
