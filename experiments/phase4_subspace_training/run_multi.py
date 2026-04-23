"""Run Phase 4 across multiple tasks and k values, summarize."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase4_subspace_training.run_train import run

OUTDIR = Path(__file__).parent / "results"
OUTDIR.mkdir(exist_ok=True)

TASKS = ["task020", "task022", "task034", "task039"]
KS    = [4, 16]    # k=32/64 OOMs on q_sparse8 targets; Phase 1 already showed k doesn't matter functionally
N_FIT = 200
STEPS = 100
BATCH_SIZE = 2
N_TRAIN = 32


def main():
    import gc
    import torch
    all_results = []
    for task in TASKS:
        for k in KS:
            print(f"\n\n============ task={task} k={k} ============\n", flush=True)
            res = run(
                held_out_task=task,
                fit_tasks=None,
                k=k,
                steps=STEPS,
                batch_size=BATCH_SIZE,
                lr_lora=1e-4,
                lr_subspace=5e-2,
                n_train=N_TRAIN,
                target_pattern="q_sparse8",
            )
            all_results.append(res)
            (OUTDIR / "phase4_multi_results.json").write_text(json.dumps(all_results, indent=2))
            gc.collect()
            torch.cuda.empty_cache()

    # Summary table
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'task':10s} {'k':4s} {'base':7s} {'held_LoRA':11s} {'lora_r16':10s} {'sub_mean':10s} {'sub_basis':10s}")
    for r in all_results:
        runs = r["runs"]
        k = r["k"]
        t = r["held_out"]
        print(f"{t:10s} {k:<4d} "
              f"{runs['base']['final_acc']:.0%}    "
              f"{runs['held_out_pretrained_lora']['final_acc']:.0%}        "
              f"{runs['lora_r16']['final_acc']:.0%}       "
              f"{runs[f'subspace_k{k}_with_mean']['final_acc']:.0%}       "
              f"{runs[f'subspace_k{k}_basis_only']['final_acc']:.0%}")


if __name__ == "__main__":
    main()
