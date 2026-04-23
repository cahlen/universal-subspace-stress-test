"""
Phase 4 — Subspace-constrained training.

The paper promises: you can train a new task by learning only a small number of
coefficients inside the universal subspace, with fewer parameters and faster
convergence than full LoRA training.

Protocol:
  1. Fit universal subspace V_k from N existing trained Mistral LoRAs.
  2. Pick a held-out task (not in fit set) with its training data.
  3. Train TWO adapters side-by-side on this task:
       (a) Standard rank-16 LoRA (baseline).  (r × d_in + d_out × r) params per layer.
       (b) Subspace adapter: learn k scalar coefficients per layer; the effective
           update is Σ_j c_j V_k[:, j] reshape, scaled.  (k params per layer).
  4. Track loss per step and task accuracy at the end.

The question is: does (b) converge as fast or better than (a) at matched step
count, with (b) using ~256× fewer trainable parameters?
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
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LORA_DIR = Path(os.environ.get("LORA_DIR", "./data/loras"))
OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(exist_ok=True)
DEVICE = "cuda"
DTYPE_MODEL = torch.bfloat16
LORA_SCALING = 2.0
LORA_RANK = 16
LORA_ALPHA = 32


def parse_lora(path: Path):
    sd = load_file(path)
    out = defaultdict(dict)
    for k, v in sd.items():
        m = re.match(r"^base_model\.model\.(.*)\.(lora_[AB])\.weight$", k)
        if m is None:
            continue
        out[m.group(1)][m.group(2)] = v.to(dtype=torch.float32)
    return {k: (v["lora_A"], v["lora_B"]) for k, v in out.items() if "lora_A" in v and "lora_B" in v}


def task_id_from_dir(d: Path):
    m = re.search(r"-(task\d+)$", d.name)
    return m.group(1) if m else ""


def parse_all(lora_dirs):
    A = defaultdict(list); B = defaultdict(list)
    for d in lora_dirs:
        p = parse_lora(d / "adapter_model.safetensors")
        for mod, (a, b) in p.items():
            A[mod].append(a); B[mod].append(b)
    complete = [m for m in A if len(A[m]) == len(lora_dirs)]
    return {m: torch.stack(A[m], dim=0) for m in complete}, {m: torch.stack(B[m], dim=0) for m in complete}, complete


# --- Build subspace on GPU via LoRA-structured Gram path ---
def build_subspace(A_stack, B_stack, max_k, chunk=50):
    N, d_out, r = B_stack.shape
    _, _, d_in = A_stack.shape
    A = A_stack.to(DEVICE)
    B = B_stack.to(DEVICE)

    BtB = torch.einsum("ior,jos->ijrs", B, B)
    AAt = torch.einsum("irp,jsp->ijrs", A, A)
    G_raw = torch.einsum("ijab,jiba->ij", BtB, AAt)
    del BtB, AAt
    torch.cuda.empty_cache()

    mean = torch.zeros(d_out, d_in, device=DEVICE)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        dW_c = torch.einsum("nor,nri->noi", B[i:j], A[i:j])
        mean.add_(dW_c.sum(0)); del dW_c
    mean.div_(N)
    mean_flat = mean.reshape(-1)
    mean_norm_sq = mean_flat.pow(2).sum()

    Xm = torch.zeros(N, device=DEVICE)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        dW_c = torch.einsum("nor,nri->noi", B[i:j], A[i:j]).reshape(j - i, -1)
        Xm[i:j] = dW_c @ mean_flat; del dW_c
    torch.cuda.empty_cache()

    G_c = G_raw - Xm.unsqueeze(1) - Xm.unsqueeze(0) + mean_norm_sq
    eigvals, eigvecs = torch.linalg.eigh(G_c)
    eigvals = torch.clamp(eigvals.flip(0), min=1e-12)
    eigvecs = eigvecs.flip(1)
    S = torch.sqrt(eigvals)
    U = eigvecs[:, :max_k].contiguous()
    ones_U = U.sum(dim=0)
    del G_raw, G_c, eigvals, eigvecs, Xm

    V_raw = torch.zeros(max_k, d_out, d_in, device=DEVICE)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        V_raw += torch.einsum("ik,ior,irp->kop", U[i:j], B[i:j], A[i:j])
    V_max = V_raw.reshape(max_k, -1).T.contiguous()   # (D, max_k)
    del V_raw
    for i in range(max_k):
        V_max[:, i].sub_(mean_flat, alpha=float(ones_U[i].item()))
    V_max.div_(S[:max_k].unsqueeze(0))
    torch.cuda.empty_cache()

    return mean.detach(), V_max.detach()      # on GPU


# --- Adapters ---
class LoRAAdapter(nn.Module):
    """Standard rank-r LoRA on a set of target modules."""
    def __init__(self, target_modules: dict[str, tuple[int, int]], rank: int = LORA_RANK, alpha: float = LORA_ALPHA):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.targets = list(target_modules.keys())
        self.A = nn.ParameterDict({
            mp.replace(".", "__"): nn.Parameter(torch.zeros(rank, din, dtype=torch.float32))
            for mp, (dout, din) in target_modules.items()
        })
        self.B = nn.ParameterDict({
            mp.replace(".", "__"): nn.Parameter(torch.zeros(dout, rank, dtype=torch.float32))
            for mp, (dout, din) in target_modules.items()
        })
        for key in self.A:
            nn.init.kaiming_uniform_(self.A[key], a=np.sqrt(5))
            nn.init.zeros_(self.B[key])

    def delta(self, mp: str) -> torch.Tensor:
        key = mp.replace(".", "__")
        return self.B[key] @ self.A[key]                         # (d_out, d_in)

    def params(self):
        return list(self.A.values()) + list(self.B.values())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.params())


class SubspaceAdapter(nn.Module):
    """Per-module k-dim coefficients in frozen universal subspace V_k with mean.

    Buffers stored in bf16 to fit in VRAM; coefficients in fp32 for stable optimization.
    """
    def __init__(self, subspaces: dict[str, tuple[torch.Tensor, torch.Tensor]], k: int, use_mean: bool = True):
        super().__init__()
        self.k = k
        self.targets = list(subspaces.keys())
        self.use_mean = use_mean
        self.shapes = {}
        for mp, (mean, V_k) in subspaces.items():
            key = mp.replace(".", "__")
            V_k_slice = V_k[:, :k].contiguous().to(dtype=torch.bfloat16)
            mean_flat = mean.reshape(-1).to(dtype=torch.bfloat16)
            self.register_buffer(f"V__{key}", V_k_slice, persistent=False)
            self.register_buffer(f"M__{key}", mean_flat, persistent=False)
            self.shapes[mp] = mean.shape
        self.c = nn.ParameterDict({
            mp.replace(".", "__"): nn.Parameter(torch.zeros(k, dtype=torch.float32))
            for mp in subspaces.keys()
        })
        self.mean_mul = 1.0 if use_mean else 0.0

    def delta(self, mp: str) -> torch.Tensor:
        key = mp.replace(".", "__")
        V = getattr(self, f"V__{key}")                # (D, k) bf16
        M = getattr(self, f"M__{key}")                # (D,)   bf16
        c = self.c[key].to(dtype=V.dtype)             # (k,)  bf16
        flat = V @ c
        if self.use_mean:
            flat = flat + M
        return flat.reshape(self.shapes[mp])

    def params(self):
        return list(self.c.values())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.params())


# --- Hook-based forward: replace W = W_base + scaling * delta via forward_pre_hook ---
def apply_adapter_hooks(model, adapter: nn.Module, scaling: float):
    handles = []
    named = dict(model.named_modules())
    targets = adapter.targets
    for mp in targets:
        mod = named.get(mp) or named.get("model." + mp)
        if mod is None:
            raise RuntimeError(f"module {mp} not found")
        if not isinstance(mod, nn.Linear):
            raise RuntimeError(f"{mp} is not Linear, is {type(mod)}")

        def make_fn(mp_local=mp, layer=mod):
            def fwd(module, inputs):
                x = inputs[0]
                d = adapter.delta(mp_local).to(x.device, dtype=x.dtype) * scaling
                # We want the layer to compute (W + d) x + b. Easiest: add d @ x to original output via a forward hook.
                # But we need pre-hook access to x and to modify output. Use forward_hook instead.
                return None
            return fwd

        # Use forward hook to add delta @ x to the original output.
        def make_hook(mp_local=mp):
            def hook(module, inp, out):
                x = inp[0]
                d = adapter.delta(mp_local).to(out.device, dtype=out.dtype) * scaling
                # x shape (..., d_in); d shape (d_out, d_in); out shape (..., d_out)
                extra = x @ d.T
                return out + extra
            return hook

        h = mod.register_forward_hook(make_hook())
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# --- Data ---
def load_task_data(task_id: str, n_train: int = 64, n_eval: int = 20):
    from huggingface_hub import HfApi
    api = HfApi()
    matches = list(api.list_datasets(author="Lots-of-LoRAs", search=task_id, limit=10))
    for m in matches:
        if task_id in m.id.split("/")[-1]:
            try:
                ds = load_dataset(m.id, split="train", streaming=True)
                all_exs = []
                for i, ex in enumerate(ds):
                    if i >= n_train + n_eval: break
                    all_exs.append(ex)
                return all_exs[:n_train], all_exs[n_train:n_train + n_eval]
            except Exception:
                continue
    return [], []


def batch_tokenize(tok, examples, max_len=1024):
    inputs = [ex["input"] for ex in examples]
    outputs = [ex["output"][0] if isinstance(ex["output"], list) else ex["output"] for ex in examples]
    full = [i + " " + o for i, o in zip(inputs, outputs)]
    # compute prompt lengths so loss is only on completion tokens
    prompt_ids = [tok(i, return_tensors=None, truncation=True, max_length=max_len).input_ids for i in inputs]
    full_enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = full_enc["input_ids"]
    attn = full_enc["attention_mask"]
    labels = input_ids.clone()
    for i, pl in enumerate(prompt_ids):
        labels[i, :min(len(pl), labels.shape[1])] = -100
    labels[attn == 0] = -100
    return input_ids, attn, labels


@torch.no_grad()
def score_task(model, tok, examples, max_new_tokens=20):
    correct = 0
    for ex in examples:
        ids = tok(ex["input"], return_tensors="pt", truncation=True, max_length=1024).input_ids.to(model.device)
        gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        pred = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
        gold = (ex["output"][0] if isinstance(ex["output"], list) else ex["output"]).strip().rstrip(".").lower()
        p = pred.split("\n")[0].rstrip(".").lower()
        if p.startswith(gold) or gold.startswith(p) or gold in p[:len(gold)+10]:
            correct += 1
    return correct / max(len(examples), 1)


def train_adapter(model, tok, adapter, train_exs, eval_exs, steps, batch_size, lr, log_every, name):
    adapter.to(DEVICE)
    opt = torch.optim.AdamW(adapter.params(), lr=lr)
    handles = apply_adapter_hooks(model, adapter, scaling=1.0)  # adapters' own scaling already baked into delta? No, we pass scaling=1.0 since adapter.delta returns what to ADD to W.
    # Note: for LoRAAdapter, delta = B@A then should be multiplied by alpha/rank = 2.0.
    # We bake the scaling into adapter in the hook.
    steps_log = []
    try:
        model.train(False)   # keep base model frozen/eval; hook still fires
        for p in model.parameters():
            p.requires_grad_(False)
        for p in adapter.params():
            p.requires_grad_(True)

        # tokenize all training examples once
        all_ids, all_attn, all_labels = batch_tokenize(tok, train_exs, max_len=1024)
        n = all_ids.shape[0]

        for step in range(steps):
            idx = torch.randperm(n)[:batch_size]
            input_ids = all_ids[idx].to(DEVICE)
            attn = all_attn[idx].to(DEVICE)
            labels = all_labels[idx].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (step + 1) % log_every == 0 or step == 0:
                steps_log.append({"step": step + 1, "loss": float(loss.item())})
                print(f"  [{name} step {step+1:4d}] loss={loss.item():.4f}", flush=True)

        # final eval
        remove_hooks(handles)
        handles = apply_adapter_hooks(model, adapter, scaling=1.0)
        acc = score_task(model, tok, eval_exs)
        print(f"  [{name}] final eval acc = {acc:.2%}", flush=True)
    finally:
        remove_hooks(handles)

    return {"log": steps_log, "final_acc": acc, "num_params": adapter.num_params()}


def build_lora_target_modules_from_sample(d_sample: Path) -> dict[str, tuple[int, int]]:
    p = parse_lora(d_sample / "adapter_model.safetensors")
    return {mod: (B.shape[0], A.shape[1]) for mod, (A, B) in p.items()}


class LoRAWithScaling(LoRAAdapter):
    def delta(self, mp: str) -> torch.Tensor:
        return super().delta(mp) * self.scaling


def select_target_modules(modules: list[str], pattern: str) -> list[str]:
    """pattern: 'all', 'qkv', 'q_only', 'q_sparse8', ..."""
    if pattern == "all":
        return modules
    if pattern == "q_only":
        return [m for m in modules if m.endswith(".q_proj")]
    if pattern == "q_sparse8":
        # keep q_proj on 8 layers evenly spaced (0, 4, 8, ..., 28)
        wanted_layers = set(range(0, 32, 4))
        out = []
        for m in modules:
            mm = re.search(r"layers\.(\d+)\.", m)
            if mm and int(mm.group(1)) in wanted_layers and m.endswith(".q_proj"):
                out.append(m)
        return out
    if pattern == "qv_sparse8":
        wanted_layers = set(range(0, 32, 4))
        out = []
        for m in modules:
            mm = re.search(r"layers\.(\d+)\.", m)
            if mm and int(mm.group(1)) in wanted_layers and (m.endswith(".q_proj") or m.endswith(".v_proj")):
                out.append(m)
        return out
    raise ValueError(f"unknown target pattern {pattern}")


def run(held_out_task: str, fit_tasks: list[str] | None, k: int, steps: int,
        batch_size: int, lr_lora: float, lr_subspace: float, n_train: int,
        target_pattern: str = "q_sparse8"):
    all_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])
    task_to_dir = {task_id_from_dir(d): d for d in all_dirs}

    if fit_tasks is None:
        # use a large pool (all except held_out)
        fit_dirs = [d for d in all_dirs if task_id_from_dir(d) != held_out_task][:200]
    else:
        fit_dirs = [task_to_dir[t] for t in fit_tasks]

    if held_out_task not in task_to_dir:
        raise RuntimeError(f"no LoRA for {held_out_task}")
    held_out_dir = task_to_dir[held_out_task]
    print(f"[plan] fit subspace on {len(fit_dirs)} LoRAs, train on task {held_out_task}", flush=True)

    # Build subspace
    print("[subspace] parsing fit LoRAs ...", flush=True)
    all_A, all_B, modules = parse_all(fit_dirs)
    modules = select_target_modules(modules, target_pattern)
    print(f"  {len(modules)} target modules (pattern={target_pattern})", flush=True)

    print(f"[subspace] computing top-{k} V per module ...", flush=True)
    subspaces: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    t0 = time.time()
    for mod in modules:
        mean, V = build_subspace(all_A[mod], all_B[mod], max_k=k)
        # bf16 to save memory; subspace directions don't need fp32
        subspaces[mod] = (mean.cpu().to(torch.bfloat16), V.cpu().to(torch.bfloat16))
        torch.cuda.empty_cache()
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    # Task data
    train_exs, eval_exs = load_task_data(held_out_task, n_train=n_train, n_eval=20)
    print(f"[data] {len(train_exs)} train, {len(eval_exs)} eval", flush=True)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "mistralai/Mistral-7B-Instruct-v0.2"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=DTYPE_MODEL, device_map="cuda")
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)

    target_modules = {m: (subspaces[m][0].shape[0], subspaces[m][0].shape[1]) for m in modules}

    results = {"held_out": held_out_task, "n_fit": len(fit_dirs), "k": k,
               "steps": steps, "batch_size": batch_size, "lr_lora": lr_lora,
               "lr_subspace": lr_subspace, "n_train": n_train,
               "runs": {}}

    # Subspace adapter (with mean)
    print("\n=== SUBSPACE (with mean) ===")
    sub_adapter = SubspaceAdapter(subspaces, k=k, use_mean=True)
    sub_adapter.mean_mul = 1.0
    r = train_adapter(model, tok, sub_adapter, train_exs, eval_exs, steps, batch_size, lr_subspace, log_every=max(1, steps // 20), name=f"subspace_k{k}")
    results["runs"][f"subspace_k{k}_with_mean"] = r
    del sub_adapter
    torch.cuda.empty_cache()

    # Subspace adapter (no mean — to test if basis alone can learn)
    print("\n=== SUBSPACE (basis only, no mean) ===")
    sub_adapter_nm = SubspaceAdapter(subspaces, k=k, use_mean=False)
    r = train_adapter(model, tok, sub_adapter_nm, train_exs, eval_exs, steps, batch_size, lr_subspace, log_every=max(1, steps // 20), name=f"subspace_k{k}_noMean")
    results["runs"][f"subspace_k{k}_basis_only"] = r
    del sub_adapter_nm
    torch.cuda.empty_cache()

    # Standard rank-16 LoRA baseline
    print("\n=== STANDARD LoRA rank-16 ===")
    lora_adapter = LoRAWithScaling(target_modules, rank=LORA_RANK, alpha=LORA_ALPHA)
    r = train_adapter(model, tok, lora_adapter, train_exs, eval_exs, steps, batch_size, lr_lora, log_every=max(1, steps // 20), name="lora_r16")
    results["runs"]["lora_r16"] = r
    del lora_adapter
    torch.cuda.empty_cache()

    # Zero-shot eval of the held-out LoRA for reference
    print("\n=== REFERENCE: held-out LoRA applied directly ===")
    sd = parse_lora(held_out_dir / "adapter_model.safetensors")
    deltas = {mod: (B @ A).detach().cpu() for mod, (A, B) in sd.items()}
    named = dict(model.named_modules())
    backup = {}
    for mp, dW in deltas.items():
        mod = named.get(mp) or named.get("model." + mp)
        if mod is None: continue
        W = mod.weight.data
        backup[mp] = W.detach().clone()
        W.add_(dW.to(W.device, dtype=W.dtype) * LORA_SCALING)
    ref_acc = score_task(model, tok, eval_exs)
    for mp, W in backup.items():
        named.get(mp, named.get("model." + mp)).weight.data.copy_(W)
    print(f"  held-out pretrained LoRA acc = {ref_acc:.2%}")
    results["runs"]["held_out_pretrained_lora"] = {"final_acc": ref_acc, "num_params": 0}

    # Base model accuracy
    base_acc = score_task(model, tok, eval_exs)
    print(f"  base (no adapter)       = {base_acc:.2%}")
    results["runs"]["base"] = {"final_acc": base_acc, "num_params": 0}

    # Summary
    print("\n=== SUMMARY ===")
    print(f"task               = {held_out_task}")
    print(f"fit pool           = {len(fit_dirs)} LoRAs")
    print(f"k                  = {k}")
    for name, r in results["runs"].items():
        p = r.get("num_params", 0)
        print(f"  {name:30s}  final acc = {r['final_acc']:.2%}   params = {p:,}")

    (OUTDIR / f"phase4_{held_out_task}_k{k}.json").write_text(json.dumps(results, indent=2))

    # Release model and subspace buffers to allow clean re-entry across multiple runs
    del model, tok, subspaces, all_A, all_B
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", default="task020")
    ap.add_argument("--fit-tasks", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-subspace", type=float, default=5e-2)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--target-pattern", default="q_sparse8", choices=["all", "q_only", "q_sparse8", "qv_sparse8"])
    args = ap.parse_args()
    run(args.held_out, args.fit_tasks, args.k, args.steps, args.batch_size, args.lr_lora, args.lr_subspace, args.n_train, args.target_pattern)
