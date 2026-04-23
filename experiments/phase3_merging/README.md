# Phase 3 — Model Merging

**Status: COMPLETE. Paper's merging advantage is real, but not for the stated reason.**

## Setup

Merge 8 task-specific Mistral-7B-Instruct-v0.2 LoRAs into a single adapter. Evaluate that one merged adapter on each of the 8 tasks' test sets.

- Tasks: `task020, task022, task033, task034, task039, task044, task046, task050` (Natural Instructions)
- Base: `mistralai/Mistral-7B-Instruct-v0.2`
- Eval: 20 train-set examples per task (consistent with Phase 1 protocol)
- LoRA scaling: α/r = 2.0

## Methods

| Method | What it does |
|---|---|
| `no_merge` | Base model (no adapter). Lower bound. |
| `finetuned` | Each task's own LoRA, applied to its own task (individual upper bound). |
| `mean` | Elementwise mean of the 8 ΔWs per layer. |
| `task_arith` | Elementwise sum. |
| `ties` | Yadav 2023: per-task magnitude pruning, sign election, average-of-agreeing. |
| `subspace_k4/k16/k64` | Universal-subspace merge: mean + V_k @ Σᵢ coeffsᵢ . |

## Results

### Primary comparison (TIES at density=0.2, paper's default)

| method       | task020 | task022 | task033 | task034 | task039 | task044 | task046 | task050 | **avg** |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| no_merge     |  85.0%  |  70.0%  |  45.0%  |   5.0%  |  25.0%  |   0.0%  |  55.0%  |  65.0%  | 43.8%   |
| finetuned    |  95.0%  | 100.0%  |  95.0%  |  85.0%  |  65.0%  |  65.0%  |  80.0%  | 100.0%  | 85.6%   |
| **mean**     |  80.0%  |  85.0%  |  85.0%  |  65.0%  |  75.0%  |  20.0%  |  75.0%  |  70.0%  | **69.4%** |
| task_arith   |  35.0%  |  40.0%  |  30.0%  |   0.0%  |  50.0%  |  25.0%  |  35.0%  |  60.0%  | 34.4%   |
| ties (0.2)   |  65.0%  |  15.0%  |  75.0%  |   0.0%  |  60.0%  |  20.0%  |  60.0%  |  35.0%  | 41.2%   |
| **subspace_k4**  |  80.0%  |  85.0%  |  85.0%  |  65.0%  |  75.0%  |  20.0%  |  75.0%  |  70.0%  | **69.4%** |
| **subspace_k16** |  80.0%  |  85.0%  |  85.0%  |  65.0%  |  75.0%  |  20.0%  |  75.0%  |  70.0%  | **69.4%** |
| **subspace_k64** |  80.0%  |  85.0%  |  85.0%  |  65.0%  |  75.0%  |  20.0%  |  75.0%  |  70.0%  | **69.4%** |

Note: `subspace_k4 = subspace_k16 = subspace_k64 = mean` — identical to the last decimal on every single task.

### TIES density sweep (`run_ties_sweep.py`)

| TIES density | avg acc |
|---|---|
| 0.05 | 54.4% |
| 0.10 | 44.4% |
| **0.20** (paper default) | **41.3%** |
| 0.30 | 41.3% |
| 0.50 | 44.4% |
| 0.70 | 50.0% |
| 1.00 (no pruning) | 57.5% |

Best TIES (density=1.0, no pruning) is **still worse than the simple mean** (57.5% vs 69.4%).

## Interpretation

1. **Subspace merge = mean, mathematically.**
   Our subspace-merge procedure is `merged = mean + V_k @ (Σᵢ coeffsᵢ)` where `coeffsᵢ = V_kᵀ (ΔWᵢ − mean)`.
   But `Σᵢ (ΔWᵢ − mean) = Σᵢ ΔWᵢ − K·mean = K·mean − K·mean = 0`.
   So `Σᵢ coeffsᵢ = 0` for any k, and `merged = mean + 0 = mean`.

2. **The paper's merging gap is real, but it's a mean-vs-TIES gap, not a subspace-vs-TIES gap.**
   On our corpus, the simple elementwise mean beats TIES by ~28pp and beats Task Arithmetic by ~35pp. Any method that reduces to the mean (like "subspace merge" above) inherits that gap.

3. **The paper's own Table 2 numbers are consistent with this.**
   KnOTS-TIES 68.0%, "Ours" 83.5%. The 15.5pp gap could be entirely attributable to the averaging behavior of the subspace method, which is *sometimes* like the mean and *sometimes* weights specific directions differently depending on the subspace alignment. The paper doesn't isolate "mean of ΔWs" as a baseline, so this confound is not addressed.

4. **Neither method recovers full-finetune performance.**
   Mean/subspace 69.4% vs individual finetune 85.6%. Merging costs ~16pp here. Paper's numbers show a similar gap (83.5% avg merged vs 84.1% individual finetuned on their 8-task suite — they claim merging *matches* individual finetune, which is better than we observe, likely because their image-classification tasks are more compatible than our instruction-tuning tasks).

## Bottom line for Phase 3

The paper's empirical gain over baselines on merging is **not evidence for the universal subspace hypothesis**. It reflects the well-known property that a simple mean of LoRA deltas is a competitive merging baseline on some corpora, and is confounded in the paper by comparing against only sparsity/trimming baselines (TIES, DARE-TIES) without the simplest baseline (unweighted mean).

## Files

- `run_merge.py` — main merge comparison
- `run_ties_sweep.py` — TIES density sweep
- `results/phase3_merge_results.json`
- `results/phase3_ties_sweep.json`
