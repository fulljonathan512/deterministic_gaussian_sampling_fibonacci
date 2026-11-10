"""Torch-based Gaussian sampling on deterministic grids.

This module mirrors the public API of :mod:`sample_gaus` but implements
all numerical operations with :mod:`torch` tensors so it can run on
GPU.  Instead of providing a covariance matrix ``cov``, you pass its
Cholesky factor ``L`` (lower triangular) such that ``cov = L @ L.T``.

The high‑level logic (grid generation, variance normalisation,
mean-correction) follows the original NumPy implementation as closely
as possible.
"""

from __future__ import annotations

from typing import Iterable

import torch
import numpy as np
from scipy.stats import norm

from grid_util import get_uniform_grid


SUPPORED_DIM = [2, 3, 4, 5, 6]
FIB_TYPES = ["ImprovedFrolov", "ClassicalFrolov", "Fibonacci", "Galois"]

def _transform_grid_gaussian_torch(
    grid: torch.Tensor,
    mu: torch.Tensor,
    chol: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch analogue of ``_transform_grid_gaussian``.

    Uses eigen‑decomposition of ``cov = chol @ chol.T`` and rescales the
    standard normal samples accordingly, then adds the mean.
    """
    
    gaus = grid
    var = torch.mean(gaus ** 2, dim=0)
    sample_count = grid.shape[0]
    if sample_count > 1:
        gaus = gaus / torch.sqrt(var)

        # scale with eigen decomposition of covariance reconstructed from chol
        cov = chol @ chol.mT
        ew, V = torch.linalg.eig(cov)
        cov = cov.numpy()
        ew, V = np.linalg.eig(cov)
        ew = torch.from_numpy(ew)
        V = torch.from_numpy(V)
        
        print(ew.dtype)
        D = torch.diag(torch.sqrt(ew))
    
    # eigenvalues are complex in general, but cov is SPD -> imaginary ~ 0
        # ew = ew.real
        # V = V.real
    
        gaus = gaus.T  # (dim, L)
        gaus = gaus.to(dtype=V.dtype, device=V.device)
        gaus = V @ D @ gaus  # (dim,dim) @ (dim,dim) @ (dim,L) -> (dim,L)
        gaus = gaus.T  # (L,dim)

        # add mean
        gaus = gaus + mu
    else:
        return mu.reshape(1,-1), None, None
    return gaus, V, D


def _transform_with_fast_cholesky_covariance_correction_torch(
    grid: torch.Tensor,
    chol: torch.Tensor,
) -> torch.Tensor | None:
    """Torch analogue of ``_transform_with_fast_cholesky_covariance_correction``.

    This follows the algorithm in [JAIF23_Frisch] V.E.  It operates in
    standardised coordinates and then applies a fast covariance
    correction.  The covariance is reconstructed as ``cov = chol @ chol.T``.
    """
    cov = chol @ chol.mT
    cov = cov.numpy()
    ew, V = np.linalg.eig(cov)
    ew = torch.from_numpy(ew)
    V = torch.from_numpy(V)
    D = torch.diag(torch.sqrt(ew))
    
    # eps = 1e-9
    # grid = torch.clamp(grid, eps, 1 - eps)
    x_std = grid
    x_std = x_std - torch.mean(x_std, dim=0, keepdim=True)
    
    # variance correction
    L = x_std.shape[0]
    v_d = 1 / L  * (x_std ** 2).sum(dim=0)
    X_stdD = x_std / torch.sqrt(v_d)
    
    # Fast Cholesky Covariance Correction
    C_stdD = 1 / L * (X_stdD.mT @ X_stdD)
    
    try:
        L_stdD = torch.linalg.cholesky(C_stdD)
        L_stdD_inv = torch.linalg.inv(L_stdD)
        differ = C_stdD - L_stdD @ L_stdD.mT
        print("Abweichung neu:")
        print(differ)
    except RuntimeError:
        # In case L_stdD non PD (for example if C is almost 0, or other numerical issues), skip the correction
        return None
    L_stdD_inv = L_stdD_inv.to(dtype=V.dtype, device=V.device)
    
    X_stdD = X_stdD.to(dtype=V.dtype, device=V.device)
    X_Gauss = V @ D @ L_stdD_inv @ X_stdD.mT  # (dim,dim) @ (dim,dim) @ (dim,dim) @ (dim,L) -> (dim,L)
    X_Gauss = X_Gauss.mT  # (L,dim)
    return X_Gauss


def _mean_correction_torch(samples: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Torch analogue of ``_mean_correction`` (just adds ``mu``)."""
    return samples + mu


def _check_parameters_torch(mu: torch.Tensor, chol: torch.Tensor, LVol: int, type: str) -> None:
    """Torch version of ``_check_parameters``.

    The covariance is not used directly; only the shape of ``chol`` is
    validated (it must be square ``(dim, dim)``).
    """
    if type not in FIB_TYPES:
        raise ValueError(f"Type {type} is not supported. Supported types are {FIB_TYPES}.")

    dim = mu.shape[0]
    if dim not in SUPPORED_DIM:
        raise ValueError(f"Dimension {dim} is not supported. Supported dimensions are {SUPPORED_DIM}.")

    if chol.shape != (dim, dim):
        raise ValueError(f"Cholesky factor shape {tuple(chol.shape)} does not match mu shape {tuple(mu.shape)}.")


def sample_gaussian_fibonacci_torch(
    mu: Iterable[float] | torch.Tensor,
    chol: torch.Tensor,
    sample_count: int = 100,
    type: str = "Fibonacci",
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, bool]:
    """Deterministic Gaussian samples using Fibonacci / Frolov grids (torch).

    This is a torch‑only counterpart of :func:`sample_gaussian_fibonacci`.
    The covariance is specified via its Cholesky factor ``chol``.

    Args:
        mu: Mean vector of length ``dim``.
        chol: Lower‑triangular Cholesky factor of the covariance matrix with
            shape ``(dim, dim)`` so that ``cov = chol @ chol.T``.
        sample_count: Number of samples ``L``.
        type: Type of deterministic grid ("Fibonacci", "ImprovedFrolov", ...).
        device: Optional torch device (e.g. "cuda" or "cpu"). If ``None``,
            inferred from ``chol``.
        dtype: Optional dtype. If ``None``, inferred from ``chol``.

    Returns:
        Tensor of shape ``(sample_count, dim)`` with Gaussian samples.
    """
    if isinstance(mu, torch.Tensor):
        mu_t = mu
    else:
        mu_t = torch.tensor(list(mu), dtype=dtype or chol.dtype, device=device or chol.device)

    if device is None:
        device = chol.device
    if dtype is None:
        dtype = chol.dtype

    mu_t = mu_t.to(device=device, dtype=dtype)
    chol = chol.to(device=device, dtype=dtype)

    _check_parameters_torch(mu_t, chol, sample_count, type)

    if sample_count < 0:
        raise ValueError("sample_count must be non‑negative")
    if sample_count == 0:
        return torch.empty((0, mu_t.shape[0]), device=device, dtype=dtype), False

    # get deterministic uniform grid in [0,1]^dim
    grid = get_uniform_grid(mu_t.shape[0], sample_count, type=type)
    eps = 1e-9
    grid = np.clip(grid, eps, 1 - eps)  # avoid inf in ppf

    # Convert grid to torch tensor in the SAME dtype/device as mu/chol to
    # avoid mixed float/double operations later on.
    grid = torch.from_numpy(norm.ppf(grid))
    grid = grid.to(device=device, dtype=dtype)
    
    isNone = False
    if sample_count > 1:
            samples = _transform_with_fast_cholesky_covariance_correction_torch(grid, chol)
            if samples is None:
                isNone = True
                # fallback to eigen decomposition method
                samples, _, _ = _transform_grid_gaussian_torch(grid, mu_t, chol)
                return samples, isNone
            samples = _mean_correction_torch(samples, mu_t)
    else:
        # For single sample, just return the mean (best point estimate)
        samples, _, _ = _transform_grid_gaussian_torch(grid, mu_t, chol)
    return samples, isNone


__all__ = [
    "sample_gaussian_fibonacci_torch",
]
