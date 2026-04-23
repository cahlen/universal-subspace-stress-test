"""
Higher-Order SVD (HOSVD) per the paper's Algorithm 1.

Stack N task matrices into a tensor of shape (N, I_1, ..., I_M), zero-center,
then compute thin SVD along each mode, keeping the top singular vectors whose
cumulative explained variance >= tau.

Paper setup (Algorithm 1 in arXiv:2512.05117): "Order 1-2 HOSVD only" — decompose
only a subset of modes. For the common case of stacked 2D weight matrices
(N, d_out, d_in), we decompose the model-axis (mode 0) to find the shared subspace
across models.
"""
from __future__ import annotations

import numpy as np
import torch


def thin_svd(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return U, S, Vh for economy SVD. Accepts any 2-D tensor."""
    return torch.linalg.svd(mat, full_matrices=False)


def explained_variance_ratio(singular_values: torch.Tensor) -> torch.Tensor:
    """Given singular values, return each component's fraction of total energy (σ²)."""
    energy = singular_values.pow(2)
    return energy / energy.sum()


def choose_rank(singular_values: torch.Tensor, tau: float = 0.99) -> int:
    """Smallest k whose cumulative explained variance >= tau."""
    evr = explained_variance_ratio(singular_values)
    cum = torch.cumsum(evr, dim=0)
    k = int(torch.searchsorted(cum, torch.tensor(tau, device=cum.device)).item()) + 1
    return min(k, singular_values.numel())


def stack_matrices(mats: list[torch.Tensor]) -> torch.Tensor:
    """Stack N matrices of identical shape into a tensor of shape (N, ...)."""
    shapes = {tuple(m.shape) for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"All matrices must have the same shape; got {shapes}")
    return torch.stack(mats, dim=0)


def model_axis_svd(
    stacked: torch.Tensor, zero_center: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten each model's weight matrix to a vector, stack into rows, SVD.

    Returns:
        mean:  (D,)  elementwise mean across models (D = product of trailing dims)
        U:     (N, r)
        S:     (r,)
        Vh:    (r, D)

    Where N = num models, D = flattened matrix size. This finds the shared subspace
    in weight space across models.
    """
    N = stacked.shape[0]
    X = stacked.reshape(N, -1)
    if zero_center:
        mean = X.mean(dim=0)
        X = X - mean
    else:
        mean = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
    U, S, Vh = thin_svd(X)
    return mean, U, S, Vh


def model_axis_spectrum(
    stacked: torch.Tensor, zero_center: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Faster variant for N << D: compute singular values via the N x N Gram matrix
    eigendecomposition. Returns (mean, singular_values, U) — we skip Vh since it
    is an (N, D) matrix that is expensive to materialize for large D.

    For reconstruction you can recompute V_k rows on demand via  V_k = (X_c.T @ U_k) / S_k.
    """
    N = stacked.shape[0]
    X = stacked.reshape(N, -1)
    if zero_center:
        mean = X.mean(dim=0)
        X = X - mean
    else:
        mean = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
    gram = X @ X.T                                      # (N, N)
    eigvals, eigvecs = torch.linalg.eigh(gram)          # ascending
    eigvals = torch.clamp(eigvals.flip(0), min=0.0)
    eigvecs = eigvecs.flip(1)
    S = torch.sqrt(eigvals)                             # σ_i
    return mean, S, eigvecs                              # eigvecs == U


def project_onto_subspace(
    stacked: torch.Tensor, Vh: torch.Tensor, mean: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Project each model's flattened weights onto top-k subspace basis, then reconstruct.
    Returns reconstructed tensor of same shape as input.
    """
    N = stacked.shape[0]
    X = stacked.reshape(N, -1)
    X_c = X - mean
    Vk = Vh[:k]                       # (k, D)
    coeffs = X_c @ Vk.T               # (N, k)
    X_recon = coeffs @ Vk + mean      # (N, D)
    return X_recon.reshape(stacked.shape)


def reconstruction_error(stacked: torch.Tensor, Vh: torch.Tensor, mean: torch.Tensor, k: int) -> float:
    """Mean Frobenius-relative error after projection onto top-k subspace."""
    recon = project_onto_subspace(stacked, Vh, mean, k)
    num = torch.linalg.norm(stacked - recon)
    den = torch.linalg.norm(stacked)
    return (num / den).item()


def gaussian_null_singular_values(N: int, D: int, sigma: float = 1.0, seed: int = 0) -> torch.Tensor:
    """
    Null baseline: an (N, D) matrix with iid N(0, sigma^2) entries has a
    well-known Marchenko-Pastur singular-value distribution. Returns the
    singular values of the zero-centered version.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, D, generator=g) * sigma
    X = X - X.mean(dim=0, keepdim=True)
    _, S, _ = thin_svd(X)
    return S
