"""
The critical functional test for the Universal-Subspace Hypothesis.

Question: if we compress a LoRA by projecting its ΔW onto the rank-k subspace
learned across ALL LoRAs, does the resulting adapter still produce outputs
close to the original?

Protocol:
  1. Pick a target LoRA (held out from the subspace-fitting set).
  2. Using the other N-1 LoRAs, compute per-layer universal subspace Vk.
  3. Project the held-out LoRA's ΔW onto Vk to get ΔW_proj.
  4. Load the base model + original ΔW weights onto the relevant linear layers.
  5. Record token logits on a set of prompts.
  6. Replace with projected weights; record again.
  7. Report (per layer k): KL(original || projected) on logits, top-1 agreement, etc.

Critical comparison: for the paper's claim to be practically useful at
k=16, the task degradation at k=16 should be small relative to no-LoRA
baseline, because the claim is that 16 directions capture task-specific
behaviour.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LORA_DIR = Path(os.environ.get("LORA_DIR", "./data/loras"))
OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(exist_ok=True)
DEVICE = "cuda"
DTYPE_SVD = torch.float32
DTYPE_MODEL = torch.bfloat16


def parse_lora(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    sd = load_file(path)
    out = defaultdict(dict)
    for k, v in sd.items():
        m = re.match(r"^base_model\.model\.(.*)\.(lora_[AB])\.weight$", k)
        if m is None:
            continue
        out[m.group(1)][m.group(2)] = v.to(dtype=DTYPE_SVD)
    return {k: (v["lora_A"], v["lora_B"]) for k, v in out.items() if "lora_A" in v and "lora_B" in v}


def compute_and_project_per_module(
    modules_A: dict[str, list[torch.Tensor]],
    modules_B: dict[str, list[torch.Tensor]],
    dW_orig: dict[str, torch.Tensor],
    ks: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, float]]:
    """
    For each module:
      - compute V_max (for k = max(ks)) on GPU
      - for every k in ks, project dW_orig[mod] onto top-k subspace
      - store only the projected ΔW (on CPU); discard V
    Returns (projections[k][mod] = dW_proj tensor, weight_rel_err_per_k)
    """
    projections: dict[int, dict[str, torch.Tensor]] = {k: {} for k in ks}
    max_k = max(ks)
    num_sq_acc = {k: 0.0 for k in ks}
    den_sq_acc = 0.0

    modules = sorted(modules_A.keys())
    for idx, mod in enumerate(modules):
        A_stack = torch.stack(modules_A[mod], dim=0).to(DEVICE)
        B_stack = torch.stack(modules_B[mod], dim=0).to(DEVICE)
        dW_stack = torch.einsum("nor,nri->noi", B_stack, A_stack)
        N = dW_stack.shape[0]
        Xf = dW_stack.reshape(N, -1)
        mean = Xf.mean(dim=0)                           # (D,)
        Xc = Xf - mean
        gram = Xc @ Xc.T
        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = torch.clamp(eigvals.flip(0), min=1e-12)
        eigvecs = eigvecs.flip(1)
        S = torch.sqrt(eigvals)
        U_max = eigvecs[:, :max_k]
        # V_max rows of size D: (D, max_k), then transpose
        V_max = (Xc.T @ U_max) / S[:max_k].unsqueeze(0)  # (D, max_k)
        # don't transpose — keep (D, max_k), dot with coefficients

        if mod in dW_orig:
            dW = dW_orig[mod].to(DEVICE).reshape(-1)
            centered = dW - mean                          # (D,)
            # coefficients in full subspace
            coeffs_full = V_max.T @ centered              # (max_k,)
            den_sq_acc += (dW.float().pow(2).sum().item())
            for k in ks:
                coeffs = coeffs_full[:k]                  # (k,)
                recon = V_max[:, :k] @ coeffs + mean      # (D,)
                recon_mat = recon.reshape(dW_orig[mod].shape).cpu()
                projections[k][mod] = recon_mat
                err_sq = (dW_orig[mod].float() - recon_mat.float()).pow(2).sum().item()
                num_sq_acc[k] += err_sq

        del A_stack, B_stack, dW_stack, Xf, Xc, gram, eigvals, eigvecs, U_max, V_max, S
        if mod in dW_orig:
            del centered, coeffs_full
        torch.cuda.empty_cache()

        if (idx + 1) % 16 == 0:
            print(f"  [sub/{idx+1}/{len(modules)}]", flush=True)

    weight_rel_err = {k: float(np.sqrt(num_sq_acc[k] / den_sq_acc)) for k in ks}
    return projections, weight_rel_err


def project_delta(dW: torch.Tensor, mean: torch.Tensor, V_k: torch.Tensor) -> torch.Tensor:
    flat = dW.reshape(-1)
    centered = flat - mean
    coeffs = V_k @ centered
    recon = V_k.T @ coeffs + mean
    return recon.reshape(dW.shape)


def original_delta(held_out_dir: Path) -> dict[str, torch.Tensor]:
    sd = parse_lora(held_out_dir / "adapter_model.safetensors")
    return {mod: (B @ A).detach().cpu() for mod, (A, B) in sd.items()}


def load_base_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"[load] {name}", flush=True)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=DTYPE_MODEL, device_map="cuda",
    )
    model.train(False)
    return model, tok


def apply_delta_inplace(model, deltas: dict[str, torch.Tensor], scaling: float = 1.0):
    backup = {}
    named = dict(model.named_modules())
    for mod_path, dW in deltas.items():
        mod = named.get(mod_path) or named.get("model." + mod_path)
        if mod is None:
            raise RuntimeError(f"Cannot locate module {mod_path}")
        W = mod.weight.data
        backup[mod_path] = W.detach().clone()
        dW_dev = dW.to(W.device, dtype=W.dtype) * scaling
        W.add_(dW_dev)
    return backup


def restore_weights(model, backup: dict[str, torch.Tensor]):
    named = dict(model.named_modules())
    for mod_path, W_orig in backup.items():
        mod = named.get(mod_path) or named.get("model." + mod_path)
        mod.weight.data.copy_(W_orig)


EVAL_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "Translate to French: Hello, how are you?",
    "Classify the sentiment: 'I hate waiting in line at the DMV.' Sentiment:",
    "Q: What is 17 * 23?\nA:",
    "The three laws of motion were formulated by",
    "Explain quantum entanglement in one sentence:",
    "Continue the poem: Roses are red, violets are",
    "List three ingredients in a classic margherita pizza:",
    "Summarize: The mitochondrion is the powerhouse of the cell.",
]


@torch.no_grad()
def measure_next_token_logits(model, tok, prompts: list[str]) -> np.ndarray:
    logits_list = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        out = model(ids).logits[0, -1]
        logits_list.append(out.float().cpu())
    return torch.stack(logits_list, dim=0).numpy()


def kl_div_from_logits(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    la = logits_a - logits_a.max(axis=-1, keepdims=True)
    lb = logits_b - logits_b.max(axis=-1, keepdims=True)
    pa = np.exp(la); pa /= pa.sum(axis=-1, keepdims=True)
    pb = np.exp(lb); pb /= pb.sum(axis=-1, keepdims=True)
    eps = 1e-12
    kl = (pa * (np.log(pa + eps) - np.log(pb + eps))).sum(axis=-1)
    return float(kl.mean())


def top1_agreement(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    return float((logits_a.argmax(-1) == logits_b.argmax(-1)).mean())


def run(max_loras: int, held_out_idx: int, ks: list[int]):
    lora_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])
    lora_dirs = lora_dirs[:max_loras]
    held_out = lora_dirs[held_out_idx]
    fit_dirs = [p for i, p in enumerate(lora_dirs) if i != held_out_idx]
    print(f"[plan] fit on {len(fit_dirs)} LoRAs, hold out: {held_out.name}", flush=True)

    modules_A = defaultdict(list); modules_B = defaultdict(list)
    for d in fit_dirs:
        parsed = parse_lora(d / "adapter_model.safetensors")
        for mod, (A, B) in parsed.items():
            modules_A[mod].append(A); modules_B[mod].append(B)
    complete = sorted([m for m in modules_A if len(modules_A[m]) == len(fit_dirs)])
    modules_A = {m: modules_A[m] for m in complete}
    modules_B = {m: modules_B[m] for m in complete}
    print(f"[fit] {len(complete)} complete modules", flush=True)

    dW_orig = original_delta(held_out)
    max_k = max(ks)
    print(f"[svd+project] fit + project held-out for ks={ks} ...", flush=True)
    t0 = time.time()
    dW_projections, weight_rel_err = compute_and_project_per_module(
        modules_A, modules_B, dW_orig, ks
    )
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    for k in ks:
        print(f"  k={k:3d}: weight-space rel err = {weight_rel_err[k]:.3f}", flush=True)

    model, tok = load_base_model()
    prompts = EVAL_PROMPTS
    scaling = 1.0

    print(f"\n[eval] measuring baseline (no LoRA) logits...", flush=True)
    logits_base = measure_next_token_logits(model, tok, prompts)

    print(f"[eval] applying ORIGINAL held-out LoRA...", flush=True)
    backup = apply_delta_inplace(model, dW_orig, scaling=scaling)
    logits_orig = measure_next_token_logits(model, tok, prompts)
    restore_weights(model, backup)

    kl_base_vs_orig = kl_div_from_logits(logits_base, logits_orig)
    top1_base_vs_orig = top1_agreement(logits_base, logits_orig)
    print(f"  base -> orig:   KL={kl_base_vs_orig:.4f}   top-1 agreement={top1_base_vs_orig:.2%}", flush=True)

    results = {
        "held_out": held_out.name,
        "n_fit": len(fit_dirs),
        "ks": ks,
        "weight_rel_err": weight_rel_err,
        "kl_base_vs_orig": kl_base_vs_orig,
        "top1_base_vs_orig": top1_base_vs_orig,
        "metrics": {},
    }

    for k in ks:
        print(f"[eval] applying PROJECTED k={k} LoRA...", flush=True)
        backup = apply_delta_inplace(model, dW_projections[k], scaling=scaling)
        logits_proj = measure_next_token_logits(model, tok, prompts)
        restore_weights(model, backup)

        kl_orig_proj = kl_div_from_logits(logits_orig, logits_proj)
        kl_base_proj = kl_div_from_logits(logits_base, logits_proj)
        t1_orig_proj = top1_agreement(logits_orig, logits_proj)
        t1_base_proj = top1_agreement(logits_base, logits_proj)

        results["metrics"][k] = {
            "kl_orig_vs_proj":  kl_orig_proj,
            "kl_base_vs_proj":  kl_base_proj,
            "top1_orig_vs_proj": t1_orig_proj,
            "top1_base_vs_proj": t1_base_proj,
            "weight_rel_err":   weight_rel_err[k],
        }
        print(f"  k={k:3d}: KL(orig||proj)={kl_orig_proj:.4f}   "
              f"top1(orig,proj)={t1_orig_proj:.2%}   "
              f"weight-err={weight_rel_err[k]:.3f}", flush=True)

    out = OUTDIR / f"functional_test_{held_out.name}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] → {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-loras", type=int, default=100)
    ap.add_argument("--held-out-idx", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 16, 32, 64, 96])
    args = ap.parse_args()
    run(args.max_loras, args.held_out_idx, args.ks)
