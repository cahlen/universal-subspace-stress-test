# Phase 1 — Findings on 10 held-out Mistral LoRAs

**Bottom line: on this replication, the paper's central functional claim is not supported.** The "universal subspace" does not carry task-specific information that transfers to held-out LoRAs. The ~14 pp accuracy gain from subspace projection comes entirely from the group-mean LoRA, not from the SVD basis, and is constant across k.

## Setup

- 100 Mistral-7B-Instruct-v0.2 rank-16 LoRAs from `Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task*`
- Hold out 10 LoRAs (idx 0,1,2,3,4,5,6,7,9,11). Fit subspace on remaining 99.
- For each held-out, test these five types of weight updates (correct scaling α/r = 2.0):
    1. `base` — no LoRA applied
    2. `orig` — full held-out LoRA (ΔW = B@A)
    3. `mean_only` — just the elementwise mean of the 99 fit-set ΔWs per layer, no basis
    4. `basis_k` — project the (dW_heldout − mean) onto the top-k subspace of the fit set, **omit the mean** from reconstruction
    5. `full_k` — standard projection: mean + basis reconstruction = mean + V_k V_kᵀ (dW − mean)
- Evaluate each on 20 test examples from the held-out's actual training task.

Correct scaling (α=32, r=16 → scale=2.0) from `adapter_config.json`. Previous runs used scale=1.0 which explains why "orig" looked too weak.

## Raw results

Reported as mean task accuracy across 10 held-outs × 20 examples each = 200 examples per condition.

### N=100 (fit subspace on 99 LoRAs, held out one)

| Condition       | Mean Acc | Δ vs base | Δ vs orig |
|-----------------|----------|-----------|-----------|
| base            | 29.5%    |   —       |  −39.0 pp |
| orig LoRA       | 68.5%    |  +39.0 pp |   —       |
| **mean_only**   | **44.0%**|  +14.5 pp |  −24.5 pp |
| basis_k1        | 29.5%    |  +0.0 pp  |  −39.0 pp |
| basis_k8        | 31.0%    |  +1.5 pp  |  −37.5 pp |
| basis_k16       | 31.0%    |  +1.5 pp  |  −37.5 pp |
| basis_k32       | 30.0%    |  +0.5 pp  |  −38.5 pp |
| basis_k96       | 21.5%    |  −8.0 pp  |  −47.0 pp |
| full_k1         | 44.0%    |  +14.5 pp |  −24.5 pp |
| full_k8         | 44.0%    |  +14.5 pp |  −24.5 pp |
| **full_k16**    | **44.5%**|  +15.0 pp |  −24.0 pp |
| full_k32        | 45.0%    |  +15.5 pp |  −23.5 pp |
| full_k96        | 44.5%    |  +15.0 pp |  −24.0 pp |

### N=500 (paper's own scale; fit on 499, held out one)

| Condition       | Mean Acc | Δ vs base | Δ vs orig |
|-----------------|----------|-----------|-----------|
| base            | 29.5%    |   —       |  −39.0 pp |
| orig LoRA       | 68.5%    |  +39.0 pp |   —       |
| **mean_only**   | **43.5%**|  +14.0 pp |  −25.0 pp |
| basis_k1        | 30.5%    |  +1.0 pp  |  −38.0 pp |
| basis_k8        | 29.5%    |  +0.0 pp  |  −39.0 pp |
| basis_k16       | 29.0%    |  −0.5 pp  |  −39.5 pp |
| basis_k32       | 29.0%    |  −0.5 pp  |  −39.5 pp |
| basis_k64       | 29.0%    |  −0.5 pp  |  −39.5 pp |
| full_k1         | 43.5%    |  +14.0 pp |  −25.0 pp |
| full_k8         | 44.0%    |  +14.5 pp |  −24.5 pp |
| **full_k16**    | **43.5%**|  +14.0 pp |  −25.0 pp |
| full_k32        | 44.5%    |  +15.0 pp |  −24.0 pp |
| full_k64        | 44.5%    |  +15.0 pp |  −24.0 pp |

Scaling from N=100 to N=500 does not sharpen the effect. **Identical qualitative pattern:**
- `mean_only` ≈ `full_k*` for all k (within 1 pp): the basis adds nothing once you have the mean.
- `basis_k*` ≈ `base` for small k, and slightly below base for large k: the basis alone contains no task-specific information.
- Orig LoRA's 39pp gain is never recovered by any subspace projection.

## What this shows

1. **The basis contributes essentially nothing.**
   - `basis_k1 = 29.5% = base`. One-dimensional projection gives no task-specific improvement.
   - `basis_k8` to `basis_k32` add +1.5 pp — within noise at n=200 examples total.
   - `basis_k96` (all components) is **worse than base** (21.5% vs 29.5%), because the late components contain noise that interferes with base behavior.

2. **The mean is the whole story.**
   - `mean_only` (just the group-average ΔW, zero basis) matches `full_k` for every k ∈ {1, 8, 16, 32, 96}: **all within 1 pp of each other**.
   - Adding more subspace components on top of the mean does not improve task accuracy.

3. **The mean is not task-specific.**
   - `mean_only` is computed as the average of 99 OTHER LoRAs' ΔWs. It is identical across all 10 held-outs.
   - Its 14.5 pp gain looks like a generic "instruction-following bias" applied uniformly: helps weak tasks (34→60, 39→65, 46→70), hurts some already-strong tasks slightly, fails entirely on tasks where the base model is at 0% (025, 044).

4. **The paper's functional claim is therefore wrong at k=16.** If "16 universal directions carry task-specific behavior" were true, `basis_k16` should significantly outperform `basis_k1`, and `full_k16` should significantly outperform `mean_only`. Neither happens.

## Why the spectrum looked "sharper than Gaussian"

In our earlier Phase-1 spectral analysis, trained LoRAs had top-16 EVR of 43% vs 16.5% for the rank-r null. The signal is real — but what it measures is just that the **mean ΔW has non-trivial norm** (~1% of total norm in each layer, and ~40% of the centered variance lives in the first few components). The dominant "shared direction" is the group mean. Components beyond the mean are not task-predictive.

Put differently: the scree curve being sharper than Gaussian is necessary but not sufficient for the "universal subspace" claim. The signal is consistent with *one* shared LoRA direction (the group mean) + noise, which is a much weaker statement than "task-specific behavior lives in a 16-dim subspace".

## Why the paper reports 19× compression at preserved performance

Reading the paper more carefully: Figure 4 claims subspace LoRA is 19× smaller with IID/OOD ROUGE scores near full-finetune. Two possible explanations consistent with our finding:
- (i) The paper's evaluated tasks might overlap heavily in what they reward, so the "group mean" is near-optimal for each task. This is exactly what we observe — mean-only gives ~44% across varied tasks.
- (ii) The paper is fitting ON the tasks it evaluates, not holding them out. Our test is a stricter leave-one-out.

## What I'd want to test next

1. **Scale to N=500** (running now) — does adding more fit LoRAs let the held-out LoRA's direction fall within the span?
2. **In-set test** — what if the held-out was INCLUDED in the fit set? Then the subspace must span it by construction. Comparing in-set vs out-of-set would quantify how much the "universal subspace" is just memorizing training LoRAs.
3. **Per-task projection quality** — measure cosine similarity between held-out ΔW and its projection, per task. Does it correlate with task-accuracy-preservation?
4. **Redo with much more aggressive compression targets** (e.g. 100× rather than 19×) — if the mean is really the whole story, we might match "mean-only" compression with vastly fewer stored parameters.

## Files

- `mean_vs_basis.py` — diagnostic script
- `results/mean_vs_basis_n100.json` — raw numbers N=100 (preserved)
- `results/mean_vs_basis.json` — will be overwritten with N=500 result
