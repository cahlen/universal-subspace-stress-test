"""
Sweep TIES density to see if a different setting beats the simple mean on our
Mistral corpus. If even tuned TIES underperforms mean, the paper's "subspace
merge > TIES" claim is actually "mean > TIES", which matches our earlier result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.phase3_merging.run_merge import (
    merge_ties, merge_mean, original_delta, task_id_from_dir, load_task_dataset,
    apply_delta, restore, score, load_base_model, LORA_SCALING, LORA_DIR,
)


def main():
    task_ids = ["task020", "task022", "task033", "task034", "task039", "task044", "task046", "task050"]
    densities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    all_dirs = sorted([p for p in LORA_DIR.iterdir() if (p / "adapter_model.safetensors").exists()])
    task_to_dir = {task_id_from_dir(d): d for d in all_dirs}
    lora_dirs = [task_to_dir[t] for t in task_ids]

    deltas = [original_delta(d) for d in lora_dirs]
    common = set(deltas[0].keys())
    for d in deltas[1:]:
        common &= set(d.keys())
    deltas = [{m: d[m] for m in common} for d in deltas]

    examples_per_task = {t: load_task_dataset(t, max_examples=20) for t in task_ids}
    model, tok = load_base_model()

    # Mean reference
    mean_merged = merge_mean(deltas)
    backup = apply_delta(model, mean_merged, LORA_SCALING)
    mean_acc = {}
    for d in lora_dirs:
        t = task_id_from_dir(d)
        mean_acc[t] = score(model, tok, examples_per_task[t])
    restore(model, backup)
    print(f"[mean] avg = {np.mean(list(mean_acc.values())):.2%}", flush=True)

    results = {"tasks": task_ids, "mean_merge": mean_acc, "ties_by_density": {}}

    for dens in densities:
        print(f"\n[ties density={dens}] ...", flush=True)
        merged = merge_ties(deltas, keep_density=dens)
        backup = apply_delta(model, merged, LORA_SCALING)
        row = {}
        for d in lora_dirs:
            t = task_id_from_dir(d)
            row[t] = score(model, tok, examples_per_task[t])
        restore(model, backup)
        avg = np.mean(list(row.values()))
        print(f"  avg = {avg:.2%}")
        results["ties_by_density"][dens] = row

    print("\n=== SUMMARY ===")
    print(f"simple mean avg:           {np.mean(list(mean_acc.values())):.2%}")
    for dens, row in results["ties_by_density"].items():
        print(f"ties density={dens:.2f} avg:   {np.mean(list(row.values())):.2%}")

    out = Path(__file__).parent / "results" / "phase3_ties_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] → {out}")


if __name__ == "__main__":
    main()
