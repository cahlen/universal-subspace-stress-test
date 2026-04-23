# Universal Weight Subspace — Replication & Validation

**Paper:** Kaushik, Chaudhari, Vaidya, Chellappa, Yuille. *The Universal Weight Subspace Hypothesis.* arXiv:2512.05117v2, Dec 2025.
**Project URL (paper):** https://toshi2k2.github.io/unisub/
**Paper URL:** https://arxiv.org/abs/2512.05117
**Started:** 2026-04-22
**Hardware:** RTX 5090 (32GB), 188GB RAM, Ubuntu 24.04, CUDA 13.1, torch 2.11.0

---

## What the paper claims

Deep neural networks trained on disjoint tasks/data/inits converge to a shared low-rank subspace **in weight space** (not representation space). Demonstrated across:

- 500 Mistral-7B LoRAs on Natural-Instructions v2 tasks
- 500 HuggingFace Vision Transformers (pretrained on various domains)
- 50 LLaMA-3-8B models
- 200 GPT-2 models, 8 Flan-T5 models
- 5 ResNet-50s trained from scratch on disjoint vision datasets
- **500 randomly-initialized ViTs (untrained!) — also share a subspace**

Key method: Stack N weight matrices, zero-center, mode-wise thin SVD (HOSVD). Keep components covering variance ≥ τ.

### Headline numbers to reproduce

| Claim | Paper's number |
|---|---|
| Rank of universal subspace (Mistral LoRAs) | ~16 directions per layer |
| Compression of 500 SDXL LoRAs | 19× smaller, equal/better CLIP |
| ViT-B/32 8-task merge: theirs vs next-best (KnOTS-TIES) | **83.5% vs 68.0%** |
| Universal ViT OOD accuracy | 87.8% (vs full-trained 91.3%) |
| Universal R50 avg across 5 datasets | 87.58% (vs full 90.89%) |

---

## Our validation plan

Each phase is designed to **falsify** a specific claim cheaply before committing more compute.

| Phase | Question | Falsifiable if... |
|---|---|---|
| 1 | Does the ~16-rank LoRA subspace replicate? | Scree curves don't show sharp knee; variance spreads uniformly |
| 2 | Is "random-init ViTs share a subspace" non-trivial? | Matched-covariance Gaussian null produces identical scree curve |
| 3 | Does subspace merging beat TIES by 15+pp? | Gap disappears on our re-run |
| 4 | Does subspace-constrained training deliver faster convergence? | Subspace-training matches or beats full finetuning in wall-clock to target accuracy |

### Concerns / potential confounds

1. **LoRA bias.** LoRAs are rank-16 by construction (typically). Finding a ~16-rank shared subspace across them is weaker evidence than it sounds.
2. **Zero-centering.** Shared mean may explain much of the "low rank" — need to separate mean contribution from basis contribution.
3. **Baseline tuning.** 68% for TIES seems low; may not be tuned. Phase 3 should test fair baselines.
4. **Null distributions.** Random matrices with iid Gaussian entries already have low-rank structure when N << flattened-dim — need a proper Marchenko-Pastur baseline.

---

## Directory layout

```
compress/
├── NOTES.md                          # This file — running log
├── README.md                         # top-level summary & repro
├── src/
│   ├── hosvd.py                      # Our HOSVD implementation (Algorithm 1)
│   ├── loaders.py                    # HF model/LoRA loading helpers
│   └── metrics.py                    # Spectral metrics, null tests
├── experiments/
│   ├── phase1_lora_spectral/
│   │   ├── README.md                 # Phase-specific findings
│   │   ├── run.py
│   │   └── results/                  # Plots, CSVs, raw eigenvalues
│   ├── phase2_random_init/
│   ├── phase3_merging/
│   └── phase4_subspace_training/
├── results/                          # Cross-phase summary plots
└── logs/                             # Run logs
```

---

## Running log (reverse chronological)

### 2026-04-22 — Phases 3 & 4 complete

**Phase 3 (merging)** — the paper's "subspace merge > TIES" claim is technically true on our corpus, but the gap is entirely the **mean's** gap, not the subspace's. Across 8 Mistral LoRAs merged into one adapter, evaluated on 8 tasks:

- Simple mean merge: **69.4%** avg accuracy (across all 8 tasks).
- `subspace_k4`, `subspace_k16`, `subspace_k64`: **69.4%** — identical to mean to the last decimal. Mathematically, merging by sum of centered coefficients always gives the mean.
- TIES at paper's default density (0.2): **41.2%**. Even best TIES (density=1.0): **57.5%**. Both worse than mean.
- Task arithmetic: 34.4%. Individual finetuned: 85.6% (upper bound).

The paper's merging advantage reduces to: *a mean of LoRA deltas beats TIES on this corpus.* The "subspace" framing is epiphenomenal.

**Phase 4 (subspace-constrained training)** — training a tiny adapter (16 coefficients per layer) that lives in the universal subspace *does* work as a cheap form of PEFT, just not because the subspace contains task structure. Across 3 held-out tasks (task034 excluded — data issue):

| Condition (k=16) | Avg acc | Params |
|---|---|---|
| base | 60.0% | 0 |
| pretrained LoRA (for reference) | 80.0% | — |
| full rank-16 LoRA (1.05M params) | **75.8%** | 131,072/layer |
| subspace (basis-only) | **70.8%** | 16/layer |
| subspace (mean + basis) | 68.3% | 16/layer |

5pp gap for 8,000× fewer parameters. Surprisingly, on simple tasks (e.g. task022 yes/no), subspace training matches or beats standard LoRA (95% vs 80%). Subspace training is regularized enough to avoid over-fitting the 32-example training set.

But `k` again doesn't matter much (k=4 vs k=16 are indistinguishable on average), and basis-only matches mean-initialized. The subspace works because it's a capacity constraint plus a cheap warm-start, not because it encodes task-specific directions.

### 2026-04-22 — main findings

**Phase 2 (random-init ViT) — paper's random-init claim not supported.**
100 freshly-initialized ViTs' weight stack is statistically indistinguishable from iid Gaussian noise of matched variance. Top-16 EVR = 16/N exactly. No low-rank subspace. See `experiments/phase2_random_init/README.md`.

**Phase 1 (trained LoRA spectra) — signal is real but much softer than paper claims.**
Using 100 Mistral-7B LoRAs from `Lots-of-LoRAs/` (the corpus the paper cites):
- Top-16 EVR = 43% across layers (median) vs null 16.5% — 2.6× over null
- Paper's "~16 directions carry majority variance" is overstated (43% ≠ majority)
- Reconstruction error at k=16: 65-78% relative. Need k≈80 for 90% variance.

**Phase 1 functional test (task accuracy) — core functional claim not supported.**
Ran 10 held-out LoRAs, projected onto the rank-k subspace of the other 99, measured real task accuracy on each held-out's own Natural Instructions task:
- Base:  29.5%   Orig: 68.5%   Mean-only: 44.0%
- **Full projection at k=1 / 8 / 16 / 32 / 64 / 96: 44.0% to 45.0% — all within 1pp of each other and of mean-only**
- **Basis-only (no mean) at k=1: 29.5% = base, zero task-specific benefit**
- **Basis-only at k=96 (full span): 21.5% — WORSE than base (adds noise)**

Interpretation: the entire gain from "universal subspace projection" is carried by the group-mean LoRA, not by any subspace structure. The mean is a fixed task-independent transformation ("generic instruction-following bias"). The subspace basis contributes no task-specific information. The paper's central claim that ~16 directions carry task-specific behavior is not supported at this scale.

Also found & fixed: `lora_alpha=32, r=16` → correct scaling is 2.0, not 1.0. Earlier runs with scale=1.0 under-applied the LoRA and looked artificially weak; this does not affect the main conclusion since all conditions use the corrected scale.

See `experiments/phase1_lora_spectral/FINDINGS.md` for full details.

**N=500 replication — CONFIRMED.** At the paper's own scale, the pattern is identical: `mean_only = full_k* = 43.5-44.5%`, `basis_k* ≈ base (29-30%) for every k ∈ {1, 8, 16, 32, 64}`. Scaling from 100→500 fit LoRAs does not sharpen the subspace. See `experiments/phase1_lora_spectral/FINDINGS.md` for both tables.

The paper's central functional claim — "~16 universal directions carry task-specific behavior" — is not supported at the paper's own sample size on the paper's own corpus.

### 2026-04-22 — kickoff

- Read paper, set up project structure.
- Chose Phases 1+2 as first priorities (cheapest + highest information value).
- Phase 2 (null test on random init) ranked most important: if it replicates the claim but so does a trivial null, we've already meaningfully critiqued the paper.
