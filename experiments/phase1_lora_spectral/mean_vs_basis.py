"""
Diagnostic: is the projected LoRA just "the mean LoRA" with noise?

For each held-out task we separately measure:
  (a) Applying ONLY the mean ΔW (from the 99-LoRA fit set)     — no basis at all
  (b) Applying projection WITHOUT mean (V_k V_k^T (dW - mean))  — basis only
  (c) Applying mean + basis projection at various k              — the full thing

If (a) ≈ (c), the "universal subspace" is just the group-average LoRA and the
basis contributes nothing at low k.

This pins down whether the paper's subspace itself carries task-specific
information, or whether the apparent benefit is entirely from the group mean.
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
from datasets import load_dataset
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LORA_DIR = Path(os.environ.get("LORA_DIR", "./data/loras"))
OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(exist_ok=True)
DEVICE = "cuda"
DTYPE_SVD = torch.float32
DTYPE_MODEL = torch.bfloat16
LORA_SCALING = 2.0


def parse_lora(path: Path):
    sd = load_file(path)
    out = defaultdict(dict)
    for k, v in sd.items():
        m = re.match(r"^base_model\.model\.(.*)\.(lora_[AB])\.weight$", k)
        if m is None:
            continue
        out[m.group(1)][m.group(2)] = v.to(dtype=DTYPE_SVD)
    return {k: (v["lora_A"], v["lora_B"]) for k, v in out.items() if "lora_A" in v and "lora_B" in v}


def parse_all(lora_dirs: list[Path]):
    modules_A = defaultdict(list); modules_B = defaultdict(list)
    for d in lora_dirs:
        p = parse_lora(d / "adapter_model.safetensors")
        for mod, (A, B) in p.items():
            modules_A[mod].append(A); modules_B[mod].append(B)
    complete = [m for m in modules_A if len(modules_A[m]) == len(lora_dirs)]
    A = {m: torch.stack(modules_A[m], dim=0) for m in complete}
    B = {m: torch.stack(modules_B[m], dim=0) for m in complete}
    return A, B, complete


def original_delta(d: Path):
    sd = parse_lora(d / "adapter_model.safetensors")
    return {mod: (B @ A).detach().cpu() for mod, (A, B) in sd.items()}


def task_id_from_dir(d: Path) -> str:
    m = re.search(r"-(task\d+)$", d.name)
    return m.group(1) if m else ""


def load_task_dataset(task_id: str, max_examples: int = 20):
    from huggingface_hub import HfApi
    api = HfApi()
    matches = list(api.list_datasets(author="Lots-of-LoRAs", search=task_id, limit=10))
    for m in matches:
        if task_id in m.id.split("/")[-1]:
            try:
                ds = load_dataset(m.id, split="train", streaming=True)
                out = []
                for i, ex in enumerate(ds):
                    if i >= max_examples: break
                    out.append(ex)
                return out
            except Exception:
                continue
    return []


def apply_delta(model, deltas, scaling):
    named = dict(model.named_modules())
    backup = {}
    for mp, dW in deltas.items():
        mod = named.get(mp) or named.get("model." + mp)
        W = mod.weight.data
        backup[mp] = W.detach().clone()
        W.add_(dW.to(W.device, dtype=W.dtype) * scaling)
    return backup


def restore(model, backup):
    named = dict(model.named_modules())
    for mp, W in backup.items():
        mod = named.get(mp) or named.get("model." + mp)
        mod.weight.data.copy_(W)


@torch.no_grad()
def score(model, tok, examples, max_new_tokens=20):
    correct = 0; total = 0
    for ex in examples:
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        new_tok = gen[0, ids.shape[1]:]
        pred = tok.decode(new_tok, skip_special_tokens=True).strip()
        gold = (ex["output"][0] if isinstance(ex["output"], list) else ex["output"]).strip().rstrip(".").lower()
        pred_norm = pred.split("\n")[0].rstrip(".").lower()
        match = pred_norm.startswith(gold) or gold.startswith(pred_norm) or gold in pred_norm[:len(gold)+10]
        if match: correct += 1
        total += 1
    return correct / max(total, 1)


def build_conditions_for_module(A_fit_cpu, B_fit_cpu, dW_heldout_cpu, ks, chunk: int = 50):
    """
    GPU computation using the low-rank LoRA structure.

    Gram matrix via structured form (never materializes the (N, D) matrix):
        <dW_i, dW_j> = tr((B_i^T B_j) (A_j A_i^T))
                     = sum_{α,β} BtB[i,j,α,β] * AtA[j,i,β,α]
    Centered Gram: G_c = G_raw - m(X) 1^T - 1 m(X)^T + N ||mean||^2

    V_max via chunked accumulation:
        W[k, o, p] = sum_i U[i,k] * (B_i A_i)[o, p]   (computed in chunks of i)
        V_max[:, k] = W[k].reshape(-1) - mean_flat * ones_U[k]    then / S[k]
    """
    max_k = max(ks)
    N = A_fit_cpu.shape[0]
    _, d_out, r = B_fit_cpu.shape
    _, _, d_in = A_fit_cpu.shape
    D = d_out * d_in

    A = A_fit_cpu.to(DEVICE)         # (N, r, d_in) ~ small
    B = B_fit_cpu.to(DEVICE)         # (N, d_out, r)

    # --- Gram matrix via structured form ---
    BtB = torch.einsum("ior,jos->ijrs", B, B)   # (N, N, r, r)  ~ N²r²*4 bytes = 256MB for N=500
    AAt = torch.einsum("irp,jsp->ijrs", A, A)   # (N, N, r, r)  A_i A_j^T at (i, j)
    # trace(BtB[i,j] @ AAt[j,i]) = sum_{α,β} BtB[i,j,α,β] * AAt[j,i,β,α]
    G_raw = torch.einsum("ijab,jiba->ij", BtB, AAt)   # (N, N)
    del BtB, AAt
    torch.cuda.empty_cache()

    # mean ΔW = (1/N) sum_i B_i A_i — compute in chunks to avoid materializing the full stack
    mean = torch.zeros(d_out, d_in, device=DEVICE)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        A_c = A[i:j]; B_c = B[i:j]
        dW_c = torch.einsum("nor,nri->noi", B_c, A_c)
        mean.add_(dW_c.sum(dim=0))
        del dW_c
    mean.div_(N)
    mean_flat = mean.reshape(-1)
    mean_norm_sq = mean_flat.pow(2).sum()

    # Xm[i] = <dW_i, mean>  -- compute via structured form:
    # <B_i A_i, M> = tr(A_i^T B_i^T M) = sum_{r} (B_i^T M)[r, :] @ A_i[r, :]
    # = trace(B_i A_i M^T)   # alt
    # Simpler: chunked dW_c @ mean_flat
    Xm = torch.zeros(N, device=DEVICE)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        A_c = A[i:j]; B_c = B[i:j]
        dW_c = torch.einsum("nor,nri->noi", B_c, A_c).reshape(j - i, -1)
        Xm[i:j] = dW_c @ mean_flat
        del dW_c
    torch.cuda.empty_cache()

    G_c = G_raw - Xm.unsqueeze(1) - Xm.unsqueeze(0) + mean_norm_sq
    eigvals, eigvecs = torch.linalg.eigh(G_c)
    eigvals = torch.clamp(eigvals.flip(0), min=1e-12)
    eigvecs = eigvecs.flip(1)
    S = torch.sqrt(eigvals)
    U_max = eigvecs[:, :max_k].contiguous()           # (N, max_k)
    ones_U = U_max.sum(dim=0)                          # (max_k,)
    del G_raw, G_c, eigvals, eigvecs, Xm

    # --- W[k, o, p] = sum_i U[i,k] * (B_i A_i)[o, p], in chunks ---
    # Stored as V_raw of shape (max_k, d_out, d_in). Memory: max_k * 4*4096²*4 ≈ max_k * 67MB.
    V_raw = torch.zeros(max_k, d_out, d_in, device=DEVICE)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        A_c = A[i:j]; B_c = B[i:j]; U_c = U_max[i:j]    # U_c: (chunk, max_k)
        # V_raw += einsum("ik, ior, irp -> kop", U_c, B_c, A_c)
        # Factor: for each k, scale B by U_c[:, k], then einsum("ior,irp->op", B_scaled, A_c).sum(0)
        # More efficient: (U_c * B_c reshape approach) but simpler is:
        V_raw += torch.einsum("ik,ior,irp->kop", U_c, B_c, A_c)
        del A_c, B_c, U_c
    # V_max[:, k] = V_raw[k].reshape(-1) - mean_flat * ones_U[k], all over S[k]
    V_max = V_raw.reshape(max_k, -1).T.contiguous()    # (D, max_k)
    del V_raw
    for i in range(max_k):
        V_max[:, i].sub_(mean_flat, alpha=float(ones_U[i].item()))
    V_max.div_(S[:max_k].unsqueeze(0))
    torch.cuda.empty_cache()

    # Project held-out
    dW_ho = dW_heldout_cpu.to(DEVICE).reshape(-1)
    coeffs_full = V_max.T @ (dW_ho - mean_flat)         # (max_k,)

    out = {}
    shape = dW_heldout_cpu.shape
    out["mean_only"] = mean.detach().cpu()
    for k in ks:
        basis = V_max[:, :k] @ coeffs_full[:k]
        out[f"basis_k{k}"] = basis.reshape(shape).detach().cpu()
        out[f"full_k{k}"] = (basis + mean_flat).reshape(shape).detach().cpu()

    del A, B, V_max, U_max, S, dW_ho, coeffs_full, mean, mean_flat, ones_U
    torch.cuda.empty_cache()
    return out


def run(max_loras, heldouts, ks, max_examples):
    lora_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])[:max_loras]
    print(f"[parse] loading {len(lora_dirs)} LoRAs ...")
    all_A, all_B, modules = parse_all(lora_dirs)
    print(f"  {len(modules)} modules")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE_MODEL, device_map="cuda")
    model.train(False)

    results = []
    N = next(iter(all_A.values())).shape[0]

    for h in heldouts:
        ho = lora_dirs[h]
        keep_mask = torch.ones(N, dtype=torch.bool); keep_mask[h] = False
        print(f"\n=== heldout {ho.name} ===")

        dW_orig = original_delta(ho)

        # Condition-keyed deltas: mapping condition -> {mod -> (d_out, d_in) tensor on CPU}
        cond_deltas: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        t0 = time.time()
        for mod in modules:
            if mod not in dW_orig:
                continue
            A_fit = all_A[mod][keep_mask]       # CPU
            B_fit = all_B[mod][keep_mask]       # CPU
            dW_ho = dW_orig[mod]                # CPU
            outs = build_conditions_for_module(A_fit, B_fit, dW_ho, ks)
            for cname, T in outs.items():
                cond_deltas[cname][mod] = T
            del A_fit, B_fit, dW_ho, outs
        print(f"  [subspaces+projections] {time.time()-t0:.1f}s")

        task_id = task_id_from_dir(ho)
        exs = load_task_dataset(task_id, max_examples=max_examples)
        if not exs:
            print(f"  skip: no dataset for {task_id}")
            continue

        base_acc = score(model, tok, exs)
        print(f"  base:         {base_acc:.2%}")

        bu = apply_delta(model, dW_orig, LORA_SCALING)
        orig_acc = score(model, tok, exs)
        restore(model, bu)
        print(f"  orig LoRA:    {orig_acc:.2%}")

        all_cond_acc: dict[str, float] = {}
        for cname in ["mean_only"] + [f"basis_k{k}" for k in ks] + [f"full_k{k}" for k in ks]:
            deltas = cond_deltas[cname]
            bu = apply_delta(model, deltas, LORA_SCALING)
            acc = score(model, tok, exs)
            restore(model, bu)
            all_cond_acc[cname] = acc
            print(f"  {cname:14s}  {acc:.2%}")

        results.append({
            "task": task_id,
            "base": base_acc, "orig": orig_acc,
            "conditions": all_cond_acc,
        })
        # Free this held-out's condition deltas before moving on
        del cond_deltas

    (OUTDIR / "mean_vs_basis.json").write_text(json.dumps(results, indent=2))

    print("\n\n=== SUMMARY ===")
    print(f"mean base:       {np.mean([r['base'] for r in results]):.2%}")
    print(f"mean orig:       {np.mean([r['orig'] for r in results]):.2%}")
    for cname in ["mean_only"] + [f"basis_k{k}" for k in ks] + [f"full_k{k}" for k in ks]:
        accs = [r["conditions"][cname] for r in results]
        print(f"mean {cname:14s}: {np.mean(accs):.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-loras", type=int, default=100)
    ap.add_argument("--heldouts", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 9, 11])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 16, 32, 64, 96])
    ap.add_argument("--max-examples", type=int, default=20)
    args = ap.parse_args()
    run(args.max_loras, args.heldouts, args.ks, args.max_examples)
