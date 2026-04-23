# Phase 1 — LoRA Spectral Replication

**Status: FIRST RESULTS IN (N=100). Signal real but much softer than paper claims. Scaling to N=500.**

## Dataset

`Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task*` — 904 LoRAs total available.
Each LoRA:
- Base: Mistral-7B-Instruct-v0.2
- Rank 16, scaling α=16, 4-bit base
- Trained on one task from Natural-Instructions v2 (Wang et al. 2022)
- Target modules: attention q_proj, k_proj, v_proj, o_proj on all 32 layers (= 128 modules; we see 96 complete)

This is the same collection cited in the paper (Brüel-Gabrielsson et al. 2024).

## Method

1. Download N adapters. Parse each to `(A, B)` pairs per target module.
2. For each module, compose `ΔW = B @ A`, shape `(d_out, d_in)`.
3. Stack the N ΔWs into `(N, d_out, d_in)`; flatten to `(N, D)`; subtract column mean.
4. Compute SV spectrum via `gram = X_c @ X_c.T` (N×N), then `σ_i = sqrt(eigvals(gram))`.
5. Null: `ΔW_null = B_null @ A_null` with A, B iid Gaussian, entrywise variance matched to empirical A, B. Same pipeline.
6. Record explained-variance ratio at top-k, reconstruction error at k=1,4,16,32,64, mean-energy fraction.

All on GPU (RTX 5090), float32. 96 layers × N=100 runs in ~10 seconds.

## Results (N=100)

### Headline numbers (medians across 96 attention layers)

| Metric | Trained LoRAs | rank-r null | ratio |
|---|---|---|---|
| top-1 EVR   | 4.17% | 1.04% | **4.0×** |
| top-16 EVR  | 43.1% | 16.5% | **2.6×** |
| top-32 EVR  | 66.0% | 32.8% | **2.0×** |
| mean-energy fraction of total norm | 1.1% | — | — |

- Reconstruction error at **k=16**: 65-78% relative (after re-adding mean)
- Reconstruction error at **k=64**: 15-25% relative
- Effective rank (99% variance) ≈ 96 (nearly all N-1 components needed)

### Interpretation

- **There is a real subspace signal.** Trained LoRAs' cumulative EVR is ~2.6× the rank-r Gaussian null at k=16 (43% vs 16%). Plots show a clear knee; the null is flat.
- **But the paper's "~16 components capture majority variance" is overstated.** 16 components capture only 43% of variance, not "majority". To get 90% you need k ≈ 80-90.
- **The 19× compression number deserves scrutiny.** In Figure 4 the paper claims 19× memory savings while preserving performance. If k=16 gives 65% reconstruction error, task performance should suffer unless the LoRA-to-model mapping is very robust to weight perturbations. Phase 3 (functional test) will check this directly.

### Which layers show the strongest subspace structure?

Top-16 EVR excess over null, per-layer:
- **Strongest:** deep `v_proj`s (layers 27-31) and layer-0 `q_proj` → excess +0.38 to +0.43
- **Weakest:** early `v_proj`s (layers 1-6) → excess +0.18 to +0.20

This is interesting — late-layer value projections have the most concentrated structure across tasks. Plausible interpretation: those layers mediate task-specific output formatting, so different tasks still reuse a common vocabulary of output-reshaping transformations.

## Plots (see `results/`)

- `scree_model_layers_*.png` — per-layer singular-value and cumulative-EVR curves, trained vs null
- `summary_distributions.png` — histograms of top-16 EVR, top-32 EVR, mean-energy fraction across all layers

## Caveats / what could still change the conclusion

1. **N=100 may be undersampling.** Paper used N=500; more LoRAs could either sharpen or flatten the spectrum. Scaling up now.
2. **The null could be stronger.** Our null draws entries of A, B iid-Gaussian; a more realistic null would match empirical A, B spectra (since trained A, B are themselves not iid). A sharper null could absorb much of our 2.6× ratio.
3. **Pre-multiplied by base.** The "universal subspace" could be an artifact of the base-model activation statistics (what gradients flow into ΔW), not of task structure. Subtracting the expected-gradient background might collapse the signal.
4. **Weight distance ≠ function distance.** The real test is what happens to LoRA outputs after projection — see Phase 3.

## Files

- `download.py` — parallel HF downloads
- `analyze.py` — main spectral analysis (Gram-matrix fast path on GPU)
- `results/phase1_report.json` — raw numbers per layer
- `results/scree_*.png` — per-layer plots
- `results/summary_distributions.png` — cross-layer distribution plot
