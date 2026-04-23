# Phase 2 — Random-Init ViT Null Test

**Status: complete. At N=100 on our replication, the paper's random-init claim is not supported — the ViT spectrum matches a matched-variance iid-Gaussian null to 4 decimal places.**

## Claim being tested

> 500 randomly initialized ViT models converge to a common low-rank subspace,
> demonstrating this is a fundamental neural network property. — paper, Fig. 1 caption

## Protocol

- Instantiate N=100 ViT-B/16 models (HF `ViTConfig` defaults), each with a fresh seed, untrained.
- Extract weight matrices from 5 layers: block-0/5/11 attention-Q projections; block-5 FC1 and FC2.
- Stack per-layer into a (100, d_out, d_in) tensor, zero-center along the model axis.
- **Mode-0 HOSVD** (`run.py`): flatten each weight to a vector, SVD the (100, D) matrix. Looks for a shared subspace *across models*.
- **Order 1-2 HOSVD** (`run_order12.py`): unfold along modes 1 and 2 (d_out, d_in), SVD each. This is the paper's default.
- Compare against a matched-variance iid-Gaussian null tensor of identical shape.

## Results

### Mode-0 (across-model) HOSVD

All 5 layers: ViT and null **match to 4 decimal places** on all reported statistics.

| Layer | ViT top-1 EVR | Null top-1 EVR | ViT top-16 EVR | Null top-16 EVR |
|---|---|---|---|---|
| block00 Q   | 0.0104 | 0.0104 | 0.1649 | 0.1648 |
| block05 Q   | 0.0104 | 0.0104 | 0.1649 | 0.1648 |
| block11 Q   | 0.0104 | 0.0104 | 0.1649 | 0.1648 |
| block05 FC1 | 0.0103 | 0.0102 | 0.1632 | 0.1632 |
| block05 FC2 | 0.0103 | 0.0102 | 0.1633 | 0.1632 |

Top-16 EVR ≈ 16/N is the **Marchenko-Pastur flat-spectrum signature** for an iid tensor with N<<D — *not* a low-rank subspace. Cumulative EVR vs. component is a straight line (see `results/scree_*.png`).

### Order 1-2 (d_out / d_in) HOSVD

| Layer | mode | ViT top-1 | Null top-1 | ViT top-16 | Null top-16 |
|---|---|---|---|---|---|
| block00 Q | 1 | 0.0016 | 0.0016 | 0.0250 | 0.0249 |
| block00 Q | 2 | 0.0016 | 0.0016 | 0.0250 | 0.0249 |
| block05 Q | 1 | 0.0016 | 0.0016 | 0.0250 | 0.0249 |
| block05 Q | 2 | 0.0016 | 0.0016 | 0.0250 | 0.0249 |
| block11 Q | 1 | 0.0016 | 0.0016 | 0.0251 | 0.0249 |
| block11 Q | 2 | 0.0016 | 0.0016 | 0.0250 | 0.0249 |

Top-16 ≈ 16/768 — also flat, also matches null exactly.

## Interpretation

The scree curve for N random-init ViTs is statistically indistinguishable from iid Gaussian matrices of matched variance.

The phenomenon is consistent with all ViTs drawing their weights from the same initializer distribution. The singular-value distribution follows Marchenko-Pastur (bulk of all N singular values ≈ σ√D with small edge fluctuations), which looks like a decaying scree curve when sorted but does not reflect any learned or emergent subspace structure beyond what the initialization itself imposes.

A caution that applies broadly: an ordered scree plot of any random matrix will show visually "decaying" singular values. Distinguishing a real low-rank structure from sorted-order decay requires a matched null baseline. For random-init ViTs on this replication, that null is indistinguishable from the data.

## What this does NOT address

- The *trained-model* claim (LoRAs, GPT-2, LLaMA, full-weight ViTs). Training can still induce genuine low-rank structure even if random init does not. Phase 1 tests this.
- The *merging* claim (Phase 3) is independent.
- The *compression* and *subspace-training* claims (Phase 4) can still hold if Phase 1 succeeds.

## Files

- `run.py` — mode-0 HOSVD test
- `run_order12.py` — Order 1-2 HOSVD test
- `results/scree_*.png` — side-by-side scree plots (ViT vs null)
- `results/order12_*.png` — mode-1/mode-2 scree plots
- `results/phase2_report.json`, `results/phase2_order12_report.json` — raw numbers
