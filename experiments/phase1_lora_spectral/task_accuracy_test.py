"""
Task-accuracy functional test.

For each held-out LoRA:
  - Load the base model
  - Fetch actual test examples from the LoRA's own Natural-Instructions task dataset
  - Record task accuracy with:
       (a) no LoRA (base)
       (b) original LoRA (ground truth performance)
       (c) LoRA projected onto rank-k universal subspace for k in {1, 8, 16, 32, 64, 96}
  - Report accuracy + generation quality

Critical: uses correct LoRA scaling α/r = 32/16 = 2.0.
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
LORA_SCALING = 2.0      # alpha / r = 32 / 16


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
    projections = {k: {} for k in ks}
    max_k = max(ks)
    num_sq = {k: 0.0 for k in ks}
    den_sq = 0.0
    for idx, mod in enumerate(sorted(modules_A.keys())):
        if mod not in dW_orig:
            continue
        A_stack = torch.stack(modules_A[mod], dim=0).to(DEVICE)
        B_stack = torch.stack(modules_B[mod], dim=0).to(DEVICE)
        dW = torch.einsum("nor,nri->noi", B_stack, A_stack)
        N = dW.shape[0]
        Xf = dW.reshape(N, -1)
        mean = Xf.mean(dim=0)
        Xc = Xf - mean
        gram = Xc @ Xc.T
        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = torch.clamp(eigvals.flip(0), min=1e-12)
        eigvecs = eigvecs.flip(1)
        S = torch.sqrt(eigvals)
        U_max = eigvecs[:, :max_k]
        V_max = (Xc.T @ U_max) / S[:max_k].unsqueeze(0)      # (D, max_k)

        dW_heldout_flat = dW_orig[mod].to(DEVICE).reshape(-1)
        centered = dW_heldout_flat - mean
        coeffs_full = V_max.T @ centered
        den_sq += (dW_orig[mod].float().pow(2).sum().item())

        for k in ks:
            coeffs_k = coeffs_full[:k]
            recon = V_max[:, :k] @ coeffs_k + mean
            recon_mat = recon.reshape(dW_orig[mod].shape).cpu()
            projections[k][mod] = recon_mat
            num_sq[k] += (dW_orig[mod].float() - recon_mat.float()).pow(2).sum().item()

        del A_stack, B_stack, dW, Xf, Xc, gram, eigvals, eigvecs, U_max, V_max, S, centered, coeffs_full
        torch.cuda.empty_cache()
    weight_rel_err = {k: float(np.sqrt(num_sq[k] / den_sq)) for k in ks}
    return projections, weight_rel_err


def original_delta(held_out_dir: Path) -> dict[str, torch.Tensor]:
    sd = parse_lora(held_out_dir / "adapter_model.safetensors")
    return {mod: (B @ A).detach().cpu() for mod, (A, B) in sd.items()}


def apply_delta_inplace(model, deltas: dict[str, torch.Tensor], scaling: float):
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


def load_base_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE_MODEL, device_map="cuda")
    model.train(False)
    return model, tok


def task_id_from_dir(d: Path) -> str:
    # e.g. "Lots-of-LoRAs__Mistral-7B-Instruct-v0.2-4b-r16-task020"
    m = re.search(r"-(task\d+)$", d.name)
    return m.group(1) if m else ""


def load_task_dataset(task_id: str, max_examples: int = 32) -> list[dict]:
    """Fetch test examples for this task. We use the train set as eval since the
    LoRA was trained on this same distribution — the functional question is
    whether projection preserves behavior on the task, not whether it generalizes
    beyond it."""
    candidates = [f"Lots-of-LoRAs/{task_id}_mctaco_span_based_question",
                  f"Lots-of-LoRAs/{task_id}"]
    # Search for the right name
    from huggingface_hub import HfApi
    api = HfApi()
    matches = list(api.list_datasets(author="Lots-of-LoRAs", search=task_id, limit=10))
    # Prefer exact prefix match
    for m in matches:
        if task_id in m.id.split("/")[-1]:
            candidates.insert(0, m.id)
            break
    for cand in candidates:
        try:
            ds = load_dataset(cand, split="train", streaming=True)
            examples = []
            for i, ex in enumerate(ds):
                if i >= max_examples:
                    break
                examples.append(ex)
            if examples:
                return examples
        except Exception:
            continue
    return []


@torch.no_grad()
def score_task(model, tok, examples: list[dict], max_new_tokens: int = 20) -> dict:
    """Run greedy generation for each example's input; compare against output[0]."""
    correct = 0
    total = 0
    outputs = []
    for ex in examples:
        prompt = ex["input"]
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
        gen = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id,
        )
        new_tokens = gen[0, ids.shape[1]:]
        new_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        gold = ex["output"][0] if isinstance(ex["output"], list) else ex["output"]
        gold_norm = gold.strip().rstrip(".").lower()
        pred_norm = new_text.strip().split("\n")[0].rstrip(".").lower()
        match = pred_norm.startswith(gold_norm) or gold_norm.startswith(pred_norm) or gold_norm in pred_norm[:len(gold_norm)+10]
        if match:
            correct += 1
        total += 1
        outputs.append({"gold": gold, "pred": new_text[:80], "match": match})
    return {"acc": correct / max(total, 1), "n": total, "examples": outputs[:5]}


def parse_all_loras(lora_dirs: list[Path]) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]], list[str]]:
    modules_A = defaultdict(list); modules_B = defaultdict(list)
    for d in lora_dirs:
        parsed = parse_lora(d / "adapter_model.safetensors")
        # fill in all known modules
        for mod, (A, B) in parsed.items():
            modules_A[mod].append(A); modules_B[mod].append(B)
    complete = [m for m in modules_A if len(modules_A[m]) == len(lora_dirs)]
    modules_A = {m: torch.stack(modules_A[m], dim=0) for m in complete}  # (N, r, d_in)
    modules_B = {m: torch.stack(modules_B[m], dim=0) for m in complete}  # (N, d_out, r)
    return modules_A, modules_B, complete


def compute_subspace_gpu_one_module(
    A_fit: torch.Tensor, B_fit: torch.Tensor, dW_heldout: torch.Tensor, ks: list[int]
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """
    Memory-conscious subspace compute. Peak mem ≈ 2 × (D × max_k) + (N × D) ≈ 3 × 6.4GB
    for 4096² layers with N=99 and max_k=96. Frees the stacked dW tensor before
    materializing V_max, and subtracts the mean correction in place.
    """
    max_k = max(ks)
    dW = torch.einsum("nor,nri->noi", B_fit, A_fit)          # (N, d_out, d_in)
    N, d_out, d_in = dW.shape
    Xf = dW.reshape(N, -1)                                   # alias of dW
    mean = Xf.mean(dim=0)                                    # (D,)
    mean_norm_sq = mean.pow(2).sum()
    Xm = Xf @ mean                                           # (N,)

    gram_raw = Xf @ Xf.T
    gram_c = gram_raw - Xm.unsqueeze(1) - Xm.unsqueeze(0) + mean_norm_sq
    del gram_raw, Xm

    eigvals, eigvecs = torch.linalg.eigh(gram_c)
    eigvals = torch.clamp(eigvals.flip(0), min=1e-12)
    eigvecs = eigvecs.flip(1)
    S = torch.sqrt(eigvals)
    U_max = eigvecs[:, :max_k].contiguous()                  # (N, max_k)
    ones_U = U_max.sum(dim=0)                                # (max_k,)
    del gram_c, eigvals, eigvecs

    # Step 1: V_max <- Xf.T @ U_max           (D, max_k)
    V_max = Xf.T @ U_max
    del dW, Xf                                               # free 6.4GB
    torch.cuda.empty_cache()

    # Step 2: subtract mean * ones_U outer product, column by column to avoid 6.4GB temp
    for i in range(max_k):
        V_max[:, i].sub_(mean, alpha=float(ones_U[i].item()))

    # Step 3: divide by S
    V_max.div_(S[:max_k].unsqueeze(0))

    # --- project held-out ---
    dWho_flat = dW_heldout.reshape(-1)
    centered = dWho_flat - mean
    coeffs_full = V_max.T @ centered                          # (max_k,)

    projections = {}
    err_sq = {}
    for k in ks:
        recon = V_max[:, :k] @ coeffs_full[:k] + mean
        recon_mat = recon.reshape(dW_heldout.shape)
        projections[k] = recon_mat.detach().cpu()
        err_sq[k] = (dW_heldout - recon_mat).float().pow(2).sum().item()

    del mean, V_max, U_max, ones_U, centered, coeffs_full, S
    torch.cuda.empty_cache()
    return projections, err_sq


def run_one_heldout_cached(
    all_A: dict[str, torch.Tensor], all_B: dict[str, torch.Tensor],
    held_out_idx: int, held_out_dir: Path,
    ks: list[int], model, tok, max_examples: int,
):
    print(f"\n{'='*60}\n[heldout] {held_out_dir.name}\n{'='*60}", flush=True)

    # Compute per-module subspaces + projections using all LoRAs except held_out_idx
    N = next(iter(all_A.values())).shape[0]
    keep_mask = torch.ones(N, dtype=torch.bool)
    keep_mask[held_out_idx] = False

    dW_orig_all = original_delta(held_out_dir)    # dict module -> (d_out, d_in) cpu tensor

    projections: dict[int, dict[str, torch.Tensor]] = {k: {} for k in ks}
    num_sq = {k: 0.0 for k in ks}; den_sq = 0.0

    t0 = time.time()
    for mod in all_A.keys():
        if mod not in dW_orig_all:
            continue
        A_fit = all_A[mod][keep_mask].to(DEVICE)
        B_fit = all_B[mod][keep_mask].to(DEVICE)
        dW_ho = dW_orig_all[mod].to(DEVICE)
        projs, errs = compute_subspace_gpu_one_module(A_fit, B_fit, dW_ho, ks)
        for k in ks:
            projections[k][mod] = projs[k]
            num_sq[k] += errs[k]
        den_sq += dW_orig_all[mod].float().pow(2).sum().item()
        del A_fit, B_fit, dW_ho
        torch.cuda.empty_cache()

    wre = {k: float(np.sqrt(num_sq[k] / den_sq)) for k in ks}
    print(f"[svd+proj] {time.time()-t0:.1f}s; weight-err by k: " +
          "  ".join(f"{k}:{wre[k]:.3f}" for k in ks), flush=True)

    task_id = task_id_from_dir(held_out_dir)
    examples = load_task_dataset(task_id, max_examples=max_examples)
    if not examples:
        print(f"  WARN: could not find task dataset for {task_id}; skipping")
        return None
    print(f"[data] {len(examples)} examples for {task_id}", flush=True)

    base = score_task(model, tok, examples)
    print(f"  base:       acc = {base['acc']:.2%}   n={base['n']}", flush=True)

    backup = apply_delta_inplace(model, dW_orig_all, scaling=LORA_SCALING)
    orig = score_task(model, tok, examples)
    restore_weights(model, backup)
    print(f"  orig LoRA:  acc = {orig['acc']:.2%}   (scaling={LORA_SCALING})", flush=True)

    proj_scores = {}
    for k in ks:
        backup = apply_delta_inplace(model, projections[k], scaling=LORA_SCALING)
        s = score_task(model, tok, examples)
        restore_weights(model, backup)
        proj_scores[k] = s
        print(f"  proj k={k:3d}: acc = {s['acc']:.2%}   weight-err={wre[k]:.3f}", flush=True)

    return {
        "held_out": held_out_dir.name,
        "task_id": task_id,
        "n_fit": int(keep_mask.sum().item()),
        "ks": ks,
        "weight_rel_err": wre,
        "base_acc": base["acc"],
        "orig_acc": orig["acc"],
        "proj_acc": {k: proj_scores[k]["acc"] for k in ks},
        "base_examples": base["examples"],
        "orig_examples": orig["examples"],
        "proj_examples": {k: proj_scores[k]["examples"] for k in ks},
    }


def run(max_loras: int, heldouts: list[int], ks: list[int], max_examples: int):
    lora_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])
    lora_dirs = lora_dirs[:max_loras]

    print(f"[parse] loading {len(lora_dirs)} LoRAs once ...", flush=True)
    t0 = time.time()
    all_A, all_B, modules = parse_all_loras(lora_dirs)
    print(f"  done in {time.time()-t0:.1f}s  modules={len(modules)}  "
          f"sample A shape={tuple(next(iter(all_A.values())).shape)}", flush=True)

    print(f"[model] loading base model once ...", flush=True)
    model, tok = load_base_model()

    all_results = []
    for h in heldouts:
        r = run_one_heldout_cached(all_A, all_B, h, lora_dirs[h], ks, model, tok, max_examples)
        if r:
            all_results.append(r)
            (OUTDIR / f"task_acc_{lora_dirs[h].name}.json").write_text(json.dumps(r, indent=2))
    (OUTDIR / "task_acc_summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\n\n=== SUMMARY ({len(all_results)} held-outs) ===")
    for r in all_results:
        ks_str = "  ".join(f"k={k}:{r['proj_acc'][k]:.2%}" for k in ks)
        print(f"  {r['task_id']:12s}  base={r['base_acc']:.2%}  orig={r['orig_acc']:.2%}   {ks_str}")
    # Means
    print("\nmean base acc:", f"{np.mean([r['base_acc'] for r in all_results]):.2%}")
    print("mean orig acc:", f"{np.mean([r['orig_acc'] for r in all_results]):.2%}")
    for k in ks:
        print(f"mean k={k:3d} proj acc:", f"{np.mean([r['proj_acc'][k] for r in all_results]):.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-loras", type=int, default=100)
    ap.add_argument("--heldouts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 16, 32, 64, 96])
    ap.add_argument("--max-examples", type=int, default=32)
    args = ap.parse_args()
    run(args.max_loras, args.heldouts, args.ks, args.max_examples)
