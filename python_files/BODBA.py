"""Modern BO-DBA attack implementation.

This module replaces the original GPyOpt-based implementation with a
BoTorch/GPyTorch optimization loop while preserving the public
``bayesian_attack(...)`` entry point used by the notebook and evaluation code.

Expected attack object interface
--------------------------------
``bayesian_attack`` expects the ``image`` argument to behave like the repo's
``python_files.Util.randomimg`` object:

* ``image.img``: original image, normally shape ``(1, H, W, C)`` in [0, 1].
* ``image.decision(candidate)``: returns True iff candidate is adversarial.
* ``image.q``: query counter incremented by ``decision``.

The implementation is intentionally TensorFlow-agnostic. It emits numpy arrays;
TensorFlow/Keras models accept numpy arrays in ``predict``.
"""

from __future__ import annotations

import colorsys
import math
import pickle
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
try:  # BoTorch exposes these re-exports in recent versions.
    from botorch.models.transforms import Normalize, Standardize
except ImportError:  # pragma: no cover - compatibility for older BoTorch.
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm.auto import tqdm

from python_files.Upsample import upsample_projection


ArrayLike = Union[np.ndarray, torch.Tensor]
LOW_DIM_NOISES = {"BICU", "BILI", "NN", "CLUSTER"}


def SAVE(fp: str, input_obj) -> None:
    """Pickle ``input_obj`` to ``fp`` for backward compatibility."""
    with open(fp, "wb+") as handle:
        pickle.dump(input_obj, handle)


def LOAD(fp: str):
    """Load a pickled object from ``fp`` for backward compatibility."""
    with open(fp, "rb+") as handle:
        return pickle.load(handle)


def millis() -> int:
    return int(round(time.time() * 1000))


def _as_numpy(value: ArrayLike) -> np.ndarray:
    """Convert a numpy / TensorFlow / torch value to a numpy array."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(np.float32, copy=False)


def _ensure_batch(image: ArrayLike) -> np.ndarray:
    arr = _as_numpy(image)
    if arr.ndim == 2:
        arr = arr[None, :, :, None]
    elif arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected image with shape (1, H, W, C); got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _norm(image: ArrayLike, image2: ArrayLike) -> Tuple[float, float]:
    y = _ensure_batch(image)
    z = _ensure_batch(image2)
    diff = (z - y).reshape(-1)
    return float(np.linalg.norm(diff, ord=2)), float(np.max(np.abs(diff)))


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.mean(rgb, axis=-1)


def normalize(vec: np.ndarray, out_min: float = 0.0, out_max: float = 1.0) -> np.ndarray:
    """Scale an array linearly to ``[out_min, out_max]``.

    Constant arrays are mapped to zeros in the requested interval to avoid
    divide-by-zero warnings during acquisition proposals near degenerate points.
    """
    vec = np.asarray(vec, dtype=np.float32)
    vmax = float(np.max(vec))
    vmin = float(np.min(vec))
    if math.isclose(vmax, vmin):
        return np.full_like(vec, out_min, dtype=np.float32)
    scaled = (vec - vmin) / (vmax - vmin)
    return (out_min + scaled * (out_max - out_min)).astype(np.float32)


def clip_image(image: ArrayLike, clip_min: float = 0.0, clip_max: float = 1.0) -> np.ndarray:
    return np.clip(_as_numpy(image), clip_min, clip_max).astype(np.float32)


def project(original_image: ArrayLike, perturbed_images: ArrayLike, alphas: float) -> np.ndarray:
    original = _ensure_batch(original_image)
    perturbed = _ensure_batch(perturbed_images)
    return ((1.0 - alphas) * original + alphas * perturbed).astype(np.float32)


def _should_stop(imgobj, query_budget: Optional[int]) -> bool:
    return query_budget is not None and getattr(imgobj, "q", 0) >= query_budget


def _decision(imgobj, candidate: np.ndarray, query_budget: Optional[int]) -> bool:
    if _should_stop(imgobj, query_budget):
        return False
    return bool(imgobj.decision(candidate.astype(np.float32, copy=False)))


def binary_search(
    perturbed_image: ArrayLike,
    imgobj,
    theta: float,
    l: str = "l2",
    query_budget: Optional[int] = None,
    max_expansion_steps: int = 20,
) -> Tuple[Optional[np.ndarray], float]:
    """Project a candidate adversarial image back toward the source image.

    If the candidate is not initially adversarial, the direction is expanded
    away from the original image, matching the behavior of the legacy code. The
    search respects ``query_budget`` when the ``imgobj`` query counter is set.
    """
    constraint = l.lower()
    if constraint in {"linf", "l_inf", "inf"}:
        constraint = "linf"
    elif constraint != "l2":
        raise ValueError("constraint must be 'l2' or 'linf'.")

    original = _ensure_batch(imgobj.img)
    candidate = clip_image(perturbed_image, 0.0, 1.0)
    candidate = _ensure_batch(candidate)

    # Expand until the candidate is adversarial or the clip box prevents motion.
    expansions = 0
    while not _decision(imgobj, candidate, query_budget):
        if _should_stop(imgobj, query_budget) or expansions >= max_expansion_steps:
            penalty = _failure_distance(original, constraint)
            return None, penalty
        previous = candidate.copy()
        candidate = clip_image(original + 2.0 * (candidate - original), 0.0, 1.0)
        expansions += 1
        if np.array_equal(previous, candidate):
            penalty = _failure_distance(original, constraint)
            return None, penalty

    low = 0.0
    high = 1.0
    # Stop once the interval is small relative to theta, as in the legacy code.
    while (high - low) / max(theta, 1e-12) > 1.0:
        if _should_stop(imgobj, query_budget):
            break
        mid = (high + low) / 2.0
        mid_image = project(original, candidate, mid)
        if _decision(imgobj, mid_image, query_budget):
            high = mid
        else:
            low = mid

    output = project(original, candidate, high)
    l2dist, linfdist = _norm(output, original)
    finaldist = l2dist if constraint == "l2" else linfdist
    return output.astype(np.float32, copy=False), finaldist


def _failure_distance(original: np.ndarray, constraint: str) -> float:
    _, height, width, channels = original.shape
    if constraint == "l2":
        return float(math.sqrt(height * width * channels))
    return 1.0


def _fbm_noise(height: int, width: int, wavelength: float, octaves: int, color_freq: float) -> np.ndarray:
    """Deterministic Perlin-like fBm fallback with no external ``noise`` package.

    The legacy code used the third-party ``noise`` extension. This replacement
    keeps the same parameter interface but uses only numpy, which makes the
    modern requirements easier to install on Python 3.11+.
    """
    wavelength = max(float(wavelength), 1.0)
    octaves = int(np.clip(round(octaves), 1, 8))
    color_freq = float(color_freq)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    field = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    amp_sum = 0.0
    for octave in range(octaves):
        scale = max(wavelength / (2.0 ** octave), 1.0)
        # A deterministic, smooth, multi-frequency field. It is not bitwise
        # equivalent to classic Perlin noise, but it has the same purpose in the
        # attack: a low-dimensional, smooth image-space perturbation family.
        phase_x = 2.0 * np.pi * (xx / scale + 0.23 * np.sin(yy / (scale * 1.7)))
        phase_y = 2.0 * np.pi * (yy / (scale * 1.3) + 0.19 * np.cos(xx / (scale * 2.1)))
        field += amplitude * (np.sin(phase_x) + np.cos(phase_y))
        amp_sum += amplitude
        amplitude *= 0.5
    field = field / max(amp_sum, 1e-12)
    field = np.sin(field * color_freq * np.pi)
    return normalize(field, 0.0, 1.0)


def perlin_noise(
    noise_scale: float,
    noise_octaves: float,
    color_freq: float,
    shape: Optional[Tuple[int, int, int]] = None,
    noise_p: float = 1.0,
    noise_l: float = 2.0,
) -> np.ndarray:
    """Create a smooth low-frequency perturbation in ``[0, 1]``.

    ``noise_p`` and ``noise_l`` are accepted for backward compatibility with the
    original signature; the deterministic numpy implementation does not use them.
    """
    del noise_p, noise_l
    if shape is None:
        raise ValueError("shape=(height, width, channels) is required in the modern implementation.")
    height, width, channels = shape
    base = _fbm_noise(height, width, noise_scale, int(round(noise_octaves)), color_freq)
    if channels == 1:
        return base[..., None].astype(np.float32)
    # Slight channel phase offsets produce color variation without adding a new
    # BO parameter.
    channel_images = []
    for ch in range(channels):
        offset = (ch / max(channels, 1)) * 0.5
        channel_images.append(normalize(np.sin((base + offset) * np.pi), 0.0, 1.0))
    return np.stack(channel_images, axis=-1).astype(np.float32)


def valid_position(size: int, x: float, y: float) -> bool:
    return 0 <= x < size and 0 <= y < size


def _odd_positive(value: float) -> int:
    value = max(1, int(round(value)))
    return value if value % 2 == 1 else value + 1


def gaborK(ksize: float, sigma: float, theta: float, lambd: float, xy_ratio: float, sides: int) -> np.ndarray:
    ksize_i = _odd_positive(ksize)
    sigma = max(float(sigma), 1e-6)
    lambd = max(float(lambd), 1e-6)
    sides = max(1, int(round(sides)))
    kernel = cv2.getGaborKernel((ksize_i, ksize_i), sigma, float(theta), lambd, float(xy_ratio), 0, ktype=cv2.CV_32F)
    for i in range(1, sides):
        kernel += cv2.getGaborKernel(
            (ksize_i, ksize_i),
            sigma,
            float(theta) + np.pi * i / sides,
            lambd,
            float(xy_ratio),
            0,
            ktype=cv2.CV_32F,
        )
    return kernel


def gabor_noise_random(
    num_kern: float,
    ksize: float,
    sigma: float,
    theta: float,
    lambd: float,
    xy_ratio: float = 1.0,
    sides: float = 1.0,
    seed: int = 0,
    shape: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Generate sparse-convolution Gabor noise in ``[0, 1]``."""
    if shape is None:
        raise ValueError("shape=(height, width, channels) is required in the modern implementation.")
    height, width, channels = shape
    grid = 20
    rng = np.random.default_rng(seed)
    kernel = gaborK(ksize, sigma, theta, lambd, xy_ratio, int(round(sides)))

    sparse = np.zeros((height, width), dtype=np.float32)
    dim_x = int(height / 2 // grid)
    dim_y = int(width / 2 // grid)
    kernels_per_cell = max(1, int(round(num_kern)))
    for i in range(-dim_x, dim_x + 1):
        for j in range(-dim_y, dim_y + 1):
            x = i * grid + height / 2 - grid / 2
            y = j * grid + width / 2 - grid / 2
            for _ in range(kernels_per_cell):
                for _attempt in range(100):
                    dx = int(rng.integers(0, grid))
                    dy = int(rng.integers(0, grid))
                    px = int(round(x + dx))
                    py = int(round(y + dy))
                    if valid_position(height, px, py) and valid_position(width, py, px):
                        sparse[px, py] = float(rng.random() * 2.0 - 1.0)
                        break

    filtered = cv2.filter2D(sparse, -1, kernel)
    normed = np.rint(normalize(filtered, 0.0, 1.0)).astype(np.float32)
    if channels == 1:
        return normed[..., None]
    return np.repeat(normed[..., None], channels, axis=-1).astype(np.float32)


rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)


def shift_hue(arr: np.ndarray, hout: float) -> np.ndarray:
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = (h + hout) % 1
    r, g, b = hsv_to_rgb(h, s, v)
    return np.dstack((r, g, b)).astype(np.float32)


def colorize(image: np.ndarray, hue: float) -> np.ndarray:
    return shift_hue(image, hue)


def _reshape_lowdim_perturbation(
    parameters: np.ndarray,
    typ: str,
    height: int,
    width: int,
    channels: int,
) -> np.ndarray:
    params = np.asarray(parameters, dtype=np.float32).reshape(1, -1)
    if params.shape[1] % channels != 0:
        raise ValueError(
            f"Low-dimensional parameter count {params.shape[1]} is not divisible by channel count {channels}."
        )
    low_dim = params.shape[1] // channels
    low_side = int(round(math.sqrt(low_dim)))
    if low_side * low_side != low_dim:
        raise ValueError(
            f"Low-dimensional per-channel size must be square; got {low_dim}. "
            "For RGB images, latent_dim=48 gives a 4x4 grid per channel."
        )
    if height != width:
        raise ValueError("Low-dimensional upsampling currently expects square images.")
    high_dim = height * width
    high = upsample_projection(typ, params, low_dim, high_dim, nchannel=channels)
    nchw = high.reshape(1, channels, height, width)
    nhwc = np.transpose(nchw, (0, 2, 3, 1))
    ptp = float(np.ptp(nhwc))
    if ptp <= 1e-12:
        return np.zeros((height, width, channels), dtype=np.float32)
    nhwc = (nhwc - float(np.min(nhwc))) / ptp
    return ((nhwc[0] - 0.5) * 2.0).astype(np.float32)


def create_distorted_image(image: ArrayLike, typ: str, epsilon: float, parameters: ArrayLike) -> np.ndarray:
    """Create an image-space perturbation from BO parameters."""
    base = _ensure_batch(image)
    _, height, width, channels = base.shape
    params = _as_numpy(parameters).reshape(-1)
    typ_key = typ.upper()

    if typ_key == "PERLIN":
        if params.size < 3:
            raise ValueError("perlin noise requires three parameters: wavelength, octave, freq_sine.")
        pert = perlin_noise(params[0], params[1], params[2], shape=(height, width, channels))
    elif typ_key == "GABOR":
        if params.size < 6:
            raise ValueError("gabor noise requires six parameters.")
        pert = gabor_noise_random(
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            sides=params[5],
            shape=(height, width, channels),
        )
    elif typ_key in LOW_DIM_NOISES:
        pert = _reshape_lowdim_perturbation(params, typ_key, height, width, channels)
    else:
        raise ValueError(f"Unsupported noise type {typ!r}. Use perlin, gabor, BICU, BILI, NN, or CLUSTER.")

    distorted = base + float(epsilon) * pert[None, ...]
    return clip_image(distorted, 0.0, 1.0)


@dataclass(frozen=True)
class SearchSpace:
    lower: torch.Tensor
    upper: torch.Tensor
    integer_indices: Tuple[int, ...]
    names: Tuple[str, ...]

    @property
    def dim(self) -> int:
        return int(self.lower.numel())

    def bounds(self) -> torch.Tensor:
        return torch.stack([self.lower, self.upper])


def _infer_channels(image_obj) -> int:
    return int(_ensure_batch(image_obj.img).shape[-1])


def _search_space(
    image_obj,
    noise: str,
    latent_dim: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
) -> SearchSpace:
    key = noise.upper()
    if key == "PERLIN":
        lower = [10.0, 1.0, 4.0]
        upper = [200.0, 4.0, 32.0]
        integers = (1,)
        names = ("wavelength", "octave", "freq_sine")
    elif key == "GABOR":
        lower = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        upper = [200.0, 40.0, 8.0, 2.0 * math.pi, 20.0, 12.0]
        integers = (0, 1, 5)
        names = ("kernels", "kernel_size", "sigma", "orientation", "scale", "sides")
    elif key in LOW_DIM_NOISES:
        channels = _infer_channels(image_obj)
        if latent_dim is None:
            latent_dim = 16 * channels
        if latent_dim % channels != 0:
            raise ValueError("latent_dim must be divisible by the image channel count.")
        per_channel = latent_dim // channels
        side = int(round(math.sqrt(per_channel)))
        if side * side != per_channel:
            raise ValueError("latent_dim / channels must be a square number.")
        lower = [-1.0] * latent_dim
        upper = [1.0] * latent_dim
        integers = ()
        names = tuple(f"z{i}" for i in range(latent_dim))
    else:
        raise ValueError(f"Unsupported noise type {noise!r}.")

    return SearchSpace(
        lower=torch.tensor(lower, dtype=dtype, device=device),
        upper=torch.tensor(upper, dtype=dtype, device=device),
        integer_indices=tuple(integers),
        names=tuple(names),
    )


def _round_discrete(x: torch.Tensor, integer_indices: Sequence[int], lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    if integer_indices:
        idx = torch.tensor(integer_indices, dtype=torch.long, device=x.device)
        x[..., idx] = x[..., idx].round()
    return torch.max(torch.min(x, upper), lower)


def _random_points(space: SearchSpace, n: int, seed: Optional[int]) -> torch.Tensor:
    if n <= 0:
        return torch.empty(0, space.dim, dtype=space.lower.dtype, device=space.lower.device)
    engine = torch.quasirandom.SobolEngine(dimension=space.dim, scramble=True, seed=seed)
    unit = engine.draw(n).to(dtype=space.lower.dtype, device=space.lower.device)
    points = space.lower + (space.upper - space.lower) * unit
    return _round_discrete(points, space.integer_indices, space.lower, space.upper)


class preddifference:
    """Callable objective wrapper kept for backward compatibility.

    The legacy code minimized distance. BoTorch maximizes acquisition functions,
    so the BO loop stores ``-distance`` as the GP observation, while this class
    continues to return positive distances.
    """

    def __init__(
        self,
        image,
        maxnorm: float,
        noise: str,
        constraint: str = "l2",
        query_budget: Optional[int] = None,
        theta: float = 0.01,
    ) -> None:
        self.image = image
        self.maxnorm = float(maxnorm)
        self.noise = noise
        self.best = float("inf")
        self.theta = float(theta)
        self.adv: Optional[np.ndarray] = None
        self.constraint = constraint
        self.query_budget = query_budget

    def func(self, parameters: ArrayLike) -> float:
        final = create_distorted_image(self.image.img, self.noise, self.maxnorm / 255.0, parameters)
        out, dist = binary_search(
            final,
            self.image,
            self.theta,
            l=self.constraint,
            query_budget=self.query_budget,
        )
        if out is not None and dist < self.best:
            self.best = float(dist)
            self.adv = out
        return float(dist)


def _fit_gp(train_x: torch.Tensor, train_y: torch.Tensor, dim: int) -> SingleTaskGP:
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=dim),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll)
    return model


def _propose_candidate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    space: SearchSpace,
    seed: Optional[int],
    num_restarts: int,
    raw_samples: int,
    maxiter: int,
) -> torch.Tensor:
    if train_x.shape[0] < 2:
        return _random_points(space, 1, seed=seed).squeeze(0)

    model = _fit_gp(train_x, train_y, space.dim)
    acq = LogExpectedImprovement(model=model, best_f=train_y.max())
    candidate, _ = optimize_acqf(
        acq_function=acq,
        bounds=space.bounds(),
        q=1,
        num_restarts=max(1, int(num_restarts)),
        raw_samples=max(2, int(raw_samples)),
        options={"maxiter": int(maxiter), "batch_limit": 5},
    )
    candidate = candidate.detach().squeeze(0)
    return _round_discrete(candidate, space.integer_indices, space.lower, space.upper)


def bayesian_attack(
    image,
    max_query: int,
    init_query: int = 5,
    noise: str = "perlin",
    max_norm: float = 16,
    constraint: str = "l2",
    *,
    seed: Optional[int] = None,
    latent_dim: Optional[int] = None,
    num_restarts: int = 5,
    raw_samples: int = 64,
    maxiter: int = 100,
    show_progress: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[List[List[float]], Optional[np.ndarray]]:
    """Run the BO-DBA attack.

    Parameters are backward-compatible with the legacy version. New keyword-only
    parameters control BoTorch candidate generation.

    Returns
    -------
    TimeHistory, adversarial_image
        ``TimeHistory`` is a list of ``[query_count, elapsed_ms]`` pairs.
        ``adversarial_image`` is the best found image or ``None`` if no
        adversarial example was found within the query budget.
    """
    if max_query <= getattr(image, "q", 0):
        return [[float(getattr(image, "q", 0)), 0.0]], None

    torch_device = torch.device(device)
    dtype = torch.double
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    space = _search_space(image, noise, latent_dim, dtype=dtype, device=torch_device)
    objective = preddifference(
        image=image,
        maxnorm=max_norm,
        noise=noise,
        constraint=constraint,
        query_budget=max_query,
    )

    train_x_parts: List[torch.Tensor] = []
    train_y_parts: List[torch.Tensor] = []
    time_history: List[List[float]] = [[float(getattr(image, "q", 0)), 0.0]]
    start = millis()

    pbar = tqdm(total=max_query, disable=not show_progress)
    pbar.n = min(getattr(image, "q", 0), max_query)
    pbar.refresh()

    def observe(candidate: torch.Tensor) -> None:
        candidate = _round_discrete(candidate, space.integer_indices, space.lower, space.upper)
        x_np = candidate.detach().cpu().numpy()
        distance = objective.func(x_np)
        train_x_parts.append(candidate.reshape(1, -1))
        # BoTorch maximizes. Distance is minimized.
        train_y_parts.append(torch.tensor([[-distance]], dtype=dtype, device=torch_device))
        elapsed = millis() - start
        time_history.append([float(getattr(image, "q", 0)), float(elapsed)])
        pbar.n = min(getattr(image, "q", 0), max_query)
        pbar.refresh()

    # Initial Sobol design.
    initial_points = _random_points(space, max(1, int(init_query)), seed=seed)
    for point in initial_points:
        if getattr(image, "q", 0) >= max_query:
            break
        observe(point)

    iteration = 0
    while getattr(image, "q", 0) < max_query:
        train_x = torch.cat(train_x_parts, dim=0)
        train_y = torch.cat(train_y_parts, dim=0)
        try:
            candidate = _propose_candidate(
                train_x=train_x,
                train_y=train_y,
                space=space,
                seed=None if seed is None else seed + 10_000 + iteration,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                maxiter=maxiter,
            )
        except Exception:
            # Surrogate fitting can fail with repeated rounded discrete points or
            # all-penalty observations. A random fallback keeps the attack alive
            # instead of aborting the entire run.
            candidate = _random_points(space, 1, seed=None if seed is None else seed + 20_000 + iteration).squeeze(0)
        observe(candidate)
        iteration += 1

    pbar.close()
    return time_history, objective.adv
