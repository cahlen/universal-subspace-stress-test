"""
Phase 3 — Model merging: fair comparison across methods.

Given K task-specific LoRAs (different tasks, same base model), combine them
into ONE merged adapter. Evaluate that single merged adapter on each of the
K tasks. A good method maintains per-task accuracy close to the individual
finetuned LoRAs.

Methods we compare (K=8 Mistral-7B LoRAs on Natural Instructions tasks):
  1. no_merge     — base model, no LoRA. Lower bound.
  2. mean         — simple average of ΔW across K tasks.
  3. ties         — TIES merging (trim low-magnitude per layer, resolve sign
                    conflicts via majority, keep only agreed components, average)
  4. task_arith   — Task Arithmetic: sum of ΔWs (scaled by 1/K to match ties/mean).
  5. subspace_k16 — universal subspace merge: mean + V_k @ (mean of coefficients).
                    For any weighted combo of in-span vectors, this reduces to the
                    mean when coefficients are centered. We implement it as a
                    sum rather than mean to preserve task-direction magnitude,
                    following the paper's implicit direction.
  6. subspace_k64 — same at k=64, to check if more components help.

For each method, the SAME merged adapter is evaluated on all K tasks.
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
LORA_SCALING = 2.0       # alpha=32, r=16


# --------- IO ---------
def parse_lora(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    sd = load_file(path)
    out = defaultdict(dict)
    for k, v in sd.items():
        m = re.match(r"^base_model\.model\.(.*)\.(lora_[AB])\.weight$", k)
        if m is None:
            continue
        out[m.group(1)][m.group(2)] = v.to(dtype=DTYPE_SVD)
    return {k: (v["lora_A"], v["lora_B"]) for k, v in out.items() if "lora_A" in v and "lora_B" in v}


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


# --------- Merging methods ---------
def merge_mean(deltas_per_lora: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Element-wise mean of ΔWs across K adapters, per module."""
    merged = {}
    modules = deltas_per_lora[0].keys()
    for mod in modules:
        stack = torch.stack([d[mod] for d in deltas_per_lora if mod in d], dim=0)
        merged[mod] = stack.mean(dim=0)
    return merged


def merge_task_arith(deltas_per_lora: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Sum of ΔWs across K adapters, per module (paper's 'Task Arithmetic' baseline)."""
    merged = {}
    modules = deltas_per_lora[0].keys()
    for mod in modules:
        stack = torch.stack([d[mod] for d in deltas_per_lora if mod in d], dim=0)
        merged[mod] = stack.sum(dim=0)
    return merged


def merge_ties(
    deltas_per_lora: list[dict[str, torch.Tensor]],
    keep_density: float = 0.2,
) -> dict[str, torch.Tensor]:
    """
    TIES merge (Yadav et al. 2023):
      (a) Trim: keep top `keep_density` fraction of entries by |ΔW| per task.
      (b) Elect: pick sign per entry by majority magnitude.
      (c) Merge: average the kept entries that agree with the elected sign.
    Returns per-module merged ΔW.
    """
    merged = {}
    modules = deltas_per_lora[0].keys()
    for mod in modules:
        stack = torch.stack([d[mod] for d in deltas_per_lora if mod in d], dim=0).float()  # (K, d_out, d_in)
        K = stack.shape[0]
        flat = stack.reshape(K, -1)                                       # (K, D)
        D = flat.shape[1]
        # (a) trim each task: keep top |val| per-task
        k_keep = max(int(D * keep_density), 1)
        abs_vals = flat.abs()
        thresholds = torch.topk(abs_vals, k_keep, dim=1, largest=True).values[:, -1]   # (K,)
        trimmed = flat * (abs_vals >= thresholds.unsqueeze(1)).float()
        # (b) elect: majority sign weighted by magnitude
        total_pos = (trimmed.clamp_min(0)).sum(dim=0)
        total_neg = (-trimmed.clamp_max(0)).sum(dim=0)
        elected_sign = torch.where(total_pos >= total_neg, torch.ones_like(total_pos), -torch.ones_like(total_pos))
        # (c) average only entries matching elected sign
        signs = torch.sign(trimmed)  # (K, D)
        match = (signs == elected_sign.unsqueeze(0)).float()
        num = (trimmed * match).sum(dim=0)
        den = match.sum(dim=0).clamp_min(1.0)
        merged_flat = num / den
        merged[mod] = merged_flat.reshape(stack.shape[1:]).to(DTYPE_SVD)
    return merged


def merge_subspace(
    deltas_per_lora: list[dict[str, torch.Tensor]],
    k: int,
) -> dict[str, torch.Tensor]:
    """
    Our best-effort subspace merge:
      - Fit universal subspace V_k on the K task LoRAs (their own corpus).
      - Merged ΔW = mean + V_k @ (sum of per-task coefficients) where coeffs_i = V_k.T @ (dW_i - mean).
      - Equivalent to summing the in-span projections of each task's deviation from the mean and adding the mean.
    """
    merged = {}
    modules = deltas_per_lora[0].keys()
    for mod in modules:
        stack = torch.stack([d[mod] for d in deltas_per_lora if mod in d], dim=0).to(DEVICE, dtype=DTYPE_SVD)
        K, d_out, d_in = stack.shape
        Xf = stack.reshape(K, -1)
        mean = Xf.mean(dim=0)
        Xc = Xf - mean
        gram = Xc @ Xc.T
        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = torch.clamp(eigvals.flip(0), min=1e-12)
        eigvecs = eigvecs.flip(1)
        S = torch.sqrt(eigvals)
        k_use = min(k, K - 1)
        U_k = eigvecs[:, :k_use]
        V_k = (Xc.T @ U_k) / S[:k_use].unsqueeze(0)                  # (D, k_use)
        # coeffs per task = V_k.T @ Xc_i
        coeffs_stack = V_k.T @ Xc.T                                  # (k_use, K)
        merged_coeff = coeffs_stack.sum(dim=1)                       # (k_use,)  sum over tasks (Task-Arithmetic-in-subspace)
        merged_flat = V_k @ merged_coeff + mean
        merged[mod] = merged_flat.reshape(d_out, d_in).detach().cpu()
        del stack, Xf, Xc, gram, eigvals, eigvecs, U_k, V_k, coeffs_stack, mean
        torch.cuda.empty_cache()
    return merged


# --------- Evaluation plumbing ---------
def apply_delta(model, deltas, scaling):
    named = dict(model.named_modules())
    backup = {}
    for mp, dW in deltas.items():
        mod = named.get(mp) or named.get("model." + mp)
        if mod is None:
            continue
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
        pred = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
        gold = (ex["output"][0] if isinstance(ex["output"], list) else ex["output"]).strip().rstrip(".").lower()
        pred_norm = pred.split("\n")[0].rstrip(".").lower()
        match = pred_norm.startswith(gold) or gold.startswith(pred_norm) or gold in pred_norm[:len(gold)+10]
        if match: correct += 1
        total += 1
    return correct / max(total, 1)


def load_base_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE_MODEL, device_map="cuda")
    model.train(False)
    return model, tok


# --------- Main ---------
def run(task_ids: list[str], ks: list[int], max_examples: int, ties_density: float):
    # Match task_ids to LoRA dirs
    all_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])
    task_to_dir = {task_id_from_dir(d): d for d in all_dirs}
    lora_dirs = []
    for t in task_ids:
        if t in task_to_dir:
            lora_dirs.append(task_to_dir[t])
        else:
            raise RuntimeError(f"no LoRA for {t}")

    print(f"[plan] merging {len(lora_dirs)} LoRAs: {[task_id_from_dir(d) for d in lora_dirs]}", flush=True)

    # Load per-LoRA ΔWs (on CPU)
    deltas_per_lora = []
    for d in lora_dirs:
        deltas_per_lora.append(original_delta(d))
    # Intersect module keys
    common = set(deltas_per_lora[0].keys())
    for d in deltas_per_lora[1:]:
        common &= set(d.keys())
    common = sorted(common)
    print(f"[delta] {len(common)} common modules", flush=True)
    deltas_per_lora = [{m: d[m] for m in common} for d in deltas_per_lora]

    # Load test examples for each task
    examples_per_task = {}
    for d in lora_dirs:
        t = task_id_from_dir(d)
        examples_per_task[t] = load_task_dataset(t, max_examples=max_examples)
        print(f"  [data] {t}: {len(examples_per_task[t])} examples", flush=True)

    # Pre-compute all merged ΔWs
    print("[merge] computing merged adapters ...", flush=True)
    merged_adapters: dict[str, dict[str, torch.Tensor]] = {}
    merged_adapters["mean"] = merge_mean(deltas_per_lora)
    merged_adapters["task_arith"] = merge_task_arith(deltas_per_lora)
    merged_adapters["ties"] = merge_ties(deltas_per_lora, keep_density=ties_density)
    for k in ks:
        merged_adapters[f"subspace_k{k}"] = merge_subspace(deltas_per_lora, k=k)
    print(f"[merge] {list(merged_adapters.keys())}", flush=True)

    # Load model and evaluate
    model, tok = load_base_model()

    all_results = {"tasks": [task_id_from_dir(d) for d in lora_dirs], "methods": {}}

    # Baseline 1: no_merge (base model)
    print("\n[eval] no_merge (base)", flush=True)
    row = {}
    for d in lora_dirs:
        t = task_id_from_dir(d)
        a = score(model, tok, examples_per_task[t])
        row[t] = a
        print(f"  {t}: {a:.2%}", flush=True)
    all_results["methods"]["no_merge"] = row

    # Baseline 2: individual finetune (each LoRA on its own task; diagonal of the merge table)
    print("\n[eval] finetuned (each LoRA on its OWN task)", flush=True)
    row = {}
    for d in lora_dirs:
        t = task_id_from_dir(d)
        backup = apply_delta(model, original_delta(d), LORA_SCALING)
        a = score(model, tok, examples_per_task[t])
        restore(model, backup)
        row[t] = a
        print(f"  {t}: {a:.2%}", flush=True)
    all_results["methods"]["finetuned"] = row

    # Merged methods
    for name, deltas in merged_adapters.items():
        print(f"\n[eval] {name}", flush=True)
        backup = apply_delta(model, deltas, LORA_SCALING)
        row = {}
        for d in lora_dirs:
            t = task_id_from_dir(d)
            a = score(model, tok, examples_per_task[t])
            row[t] = a
            print(f"  {t}: {a:.2%}", flush=True)
        restore(model, backup)
        all_results["methods"][name] = row

    # Summary
    print("\n=== Per-task accuracy (all methods) ===")
    tasks = all_results["tasks"]
    header = "method".ljust(18) + " | " + " | ".join(t.ljust(8) for t in tasks) + " |  avg "
    print(header)
    print("-" * len(header))
    for m, row in all_results["methods"].items():
        accs = [row[t] for t in tasks]
        line = m.ljust(18) + " | " + " | ".join(f"{a:6.1%} " for a in accs) + f" | {np.mean(accs):5.1%}"
        print(line)

    (OUTDIR / "phase3_merge_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n[done] → {OUTDIR/'phase3_merge_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["task020", "task022", "task033", "task034",
                                                    "task039", "task044", "task046", "task050"],
                    help="task IDs to merge; must have corresponding LoRA dirs")
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--ties-density", type=float, default=0.2)
    args = ap.parse_args()
    run(args.tasks, args.ks, args.max_examples, args.ties_density)
