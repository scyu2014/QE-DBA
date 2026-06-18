"""Modernized data and perturbation utilities for QE-DBA.

The main compatibility fix here is replacing removed ``torch.irfft`` calls with
``torch.fft`` APIs. The multi-channel transform now follows the file comment and
performs the inverse FFT independently for each channel over spatial dimensions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_mnist_data(root: str = "./data/mnist", download: bool = True):
    return dsets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=download)


def load_cifar10_data(root: str = "./data/cifar10-py", download: bool = True):
    return dsets.CIFAR10(root, download=download, train=False, transform=transforms.ToTensor())


def load_imagenet_data(root: str = "../../../../../elm/ILSVRC2012/val/", size1: int = 256, size2: int = 224):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"ImageNet validation path does not exist: {root}")
    return dsets.ImageFolder(
        str(root_path),
        transform=transforms.Compose([
            transforms.Resize(size1),
            transforms.CenterCrop(size2),
            transforms.ToTensor(),
        ]),
    )


def proj(pert: torch.Tensor, eps: float, inf_norm: bool, discrete: bool) -> torch.Tensor:
    """Project perturbations into an L_inf or L_2 epsilon ball."""
    if inf_norm:
        return eps * pert.sign() if discrete else pert.clamp(-eps, eps)

    flat = pert.reshape(pert.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1).clamp_min(1e-12)
    factors = torch.clamp(norms / float(eps), min=1.0)
    return pert / factors.view(-1, 1, 1, 1)


def latent_proj(pert: torch.Tensor, eps: float) -> torch.Tensor:
    """Project latent variables into an L_2 epsilon ball."""
    norms = torch.linalg.vector_norm(pert, ord=2, dim=1).clamp_min(1e-12)
    factors = torch.clamp(norms / float(eps), min=1.0)
    return pert / factors.view(-1, 1)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _square_dim_from_real_imag(num_values: int, divisor: int = 1) -> int:
    dim = int(round(math.sqrt(num_values / divisor)))
    if dim * dim * divisor != num_values:
        raise ValueError(f"Latent dimension {num_values} is incompatible with divisor {divisor}.")
    return dim


def fft_transform(pert: torch.Tensor) -> torch.Tensor:
    """Single-channel inverse FFT transform for MNIST-sized perturbations.

    ``pert`` stores complex coefficients using the legacy convention where the
    last latent dimension alternates real/imaginary values via a trailing size-2
    dimension after reshape.
    """
    batch = pert.shape[0]
    device = pert.device
    dtype = pert.dtype
    t_dim = _square_dim_from_real_imag(pert.shape[1], divisor=2)

    coeffs = pert.reshape(batch, t_dim, t_dim, 2).contiguous()
    complex_block = torch.view_as_complex(coeffs)
    spectrum = torch.zeros(batch, 28, 28, dtype=_complex_dtype(dtype), device=device)
    spectrum[:, :t_dim, :t_dim] = complex_block
    spatial = torch.fft.ifft2(spectrum, dim=(-2, -1), norm="ortho").real
    return spatial.unsqueeze(1).to(dtype=dtype)


def fft_transform_mc(
    pert: torch.Tensor,
    dataset_size: int,
    channel: int,
    cosine: bool,
    sine: bool,
) -> torch.Tensor:
    """Multi-channel low-frequency inverse FFT transform.

    Unlike the removed ``torch.irfft(..., signal_ndim=3)`` call, this function
    intentionally transforms only the spatial dimensions for each channel.
    """
    if not cosine and not sine:
        raise ValueError("At least one of cosine or sine must be True.")

    batch = pert.shape[0]
    device = pert.device
    dtype = pert.dtype
    spectrum = torch.zeros(batch, channel, dataset_size, dataset_size, dtype=_complex_dtype(dtype), device=device)

    if cosine and sine:
        t_dim = _square_dim_from_real_imag(pert.shape[1], divisor=channel * 2)
        coeffs = pert.reshape(batch, channel, t_dim, t_dim, 2).contiguous()
        block = torch.view_as_complex(coeffs)
    else:
        t_dim = _square_dim_from_real_imag(pert.shape[1], divisor=channel)
        real_values = pert.reshape(batch, channel, t_dim, t_dim)
        zeros = torch.zeros_like(real_values)
        if cosine:
            block = torch.complex(real_values, zeros)
        else:
            block = torch.complex(zeros, real_values)

    spectrum[:, :, :t_dim, :t_dim] = block
    spatial = torch.fft.ifft2(spectrum, dim=(-2, -1), norm="ortho").real
    return spatial.to(dtype=dtype)


def transform(pert: torch.Tensor, dset: str, arch: str, cosine: bool, sine: bool) -> torch.Tensor:
    dset_key = dset.lower()
    if dset_key == "cifar":
        return fft_transform_mc(pert, 32, 3, cosine, sine)
    if dset_key == "imagenet":
        size = 299 if arch == "Inception" else 224
        return fft_transform_mc(pert, size, 3, cosine, sine)
    if dset_key == "mnist":
        return fft_transform(pert)
    raise ValueError("dset must be CIFAR, imagenet, or mnist.")
