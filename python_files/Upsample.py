#!/usr/bin/env python3
"""Modern up/down-sampling helpers for BO-DBA low-dimensional attacks.

The original implementation mixed NHWC and NCHW flattening for different modes
and used ``.data.numpy()``. This version keeps a consistent contract:

* input ``X_low`` / ``X_high`` is flattened channel-first per sample;
* output is flattened channel-first per sample;
* torch tensors are detached safely before conversion to numpy.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

Mode = Literal["CLUSTER", "BILI", "NN", "BICU"]


def _square_side(value: int, name: str) -> int:
    side = int(round(math.sqrt(int(value))))
    if side * side != int(value):
        raise ValueError(f"{name} must be a square number; got {value}.")
    return side


def _mode(dim_reduction: str) -> Mode:
    mode = dim_reduction.upper()
    if mode not in {"CLUSTER", "BILI", "NN", "BICU"}:
        raise ValueError("dim_reduction must be one of CLUSTER, BILI, NN, or BICU.")
    return mode  # type: ignore[return-value]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def upsample_projection(
    dim_reduction: str,
    X_low,
    low_dim: int,
    high_dim: int,
    nchannel: int = 1,
    align_corners: bool = True,
) -> np.ndarray:
    """Upsample flattened low-dimensional perturbations.

    Parameters
    ----------
    dim_reduction:
        ``CLUSTER``, ``BILI``, ``NN``, or ``BICU``.
    X_low:
        Array with shape ``(n, low_dim * nchannel)`` using channel-first
        flattening: all pixels of channel 0, then channel 1, etc.
    low_dim, high_dim:
        Number of pixels per channel in the low/high representation.
    nchannel:
        Number of channels.

    Returns
    -------
    np.ndarray
        Shape ``(n, high_dim * nchannel)`` with channel-first flattening.
    """
    mode = _mode(dim_reduction)
    X_low = _to_numpy(X_low)
    if X_low.ndim == 1:
        X_low = X_low.reshape(1, -1)

    n = X_low.shape[0]
    low_side = _square_side(low_dim, "low_dim")
    high_side = _square_side(high_dim, "high_dim")
    expected = low_dim * nchannel
    if X_low.shape[1] != expected:
        raise ValueError(f"Expected X_low.shape[1] == {expected}; got {X_low.shape[1]}.")

    low = X_low.reshape(n, nchannel, low_side, low_side)

    if mode == "CLUSTER":
        scale = int(math.ceil(high_side / low_side))
        high = np.repeat(np.repeat(low, scale, axis=2), scale, axis=3)
        high = high[:, :, :high_side, :high_side]
        return high.reshape(n, high_dim * nchannel).astype(np.float32, copy=False)

    if mode == "BILI":
        interpolate_mode = "bilinear"
        kwargs = {"align_corners": align_corners}
    elif mode == "NN":
        interpolate_mode = "nearest"
        kwargs = {}
    else:  # BICU
        interpolate_mode = "bicubic"
        kwargs = {"align_corners": align_corners}

    tensor = torch.as_tensor(low, dtype=torch.float32)
    with torch.no_grad():
        high_tensor = F.interpolate(tensor, size=(high_side, high_side), mode=interpolate_mode, **kwargs)
    return high_tensor.detach().cpu().numpy().reshape(n, high_dim * nchannel).astype(np.float32, copy=False)


def downsample_projection(
    dim_reduction: str,
    X_high,
    low_dim: int,
    high_dim: int,
    nchannel: int = 1,
    align_corners: bool = True,
) -> np.ndarray:
    """Downsample flattened high-dimensional perturbations.

    Contract mirrors :func:`upsample_projection`: channel-first flattening in and
    channel-first flattening out.
    """
    mode = _mode(dim_reduction)
    X_high = _to_numpy(X_high)
    if X_high.ndim == 1:
        X_high = X_high.reshape(1, -1)

    n = X_high.shape[0]
    low_side = _square_side(low_dim, "low_dim")
    high_side = _square_side(high_dim, "high_dim")
    expected = high_dim * nchannel
    if X_high.shape[1] != expected:
        raise ValueError(f"Expected X_high.shape[1] == {expected}; got {X_high.shape[1]}.")

    high = X_high.reshape(n, nchannel, high_side, high_side)

    if mode == "CLUSTER":
        scale = int(math.ceil(high_side / low_side))
        padded_side = low_side * scale
        if padded_side != high_side:
            pad_h = padded_side - high_side
            high = np.pad(high, ((0, 0), (0, 0), (0, pad_h), (0, pad_h)), mode="edge")
        reshaped = high.reshape(n, nchannel, low_side, scale, low_side, scale)
        low = reshaped.mean(axis=(3, 5))
        return low.reshape(n, low_dim * nchannel).astype(np.float32, copy=False)

    # Use area interpolation for the inverse of the smooth modes. It is stable
    # and does not need align_corners handling.
    tensor = torch.as_tensor(high, dtype=torch.float32)
    with torch.no_grad():
        low_tensor = F.interpolate(tensor, size=(low_side, low_side), mode="area")
    return low_tensor.detach().cpu().numpy().reshape(n, low_dim * nchannel).astype(np.float32, copy=False)
