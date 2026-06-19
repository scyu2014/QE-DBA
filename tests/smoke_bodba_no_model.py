from pathlib import Path
import sys

# Let this script import python_files when run from tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from python_files.BODBA import bayesian_attack, create_distorted_image


class DummyImage:
    """
    Minimal stand-in for python_files.Util.randomimg.

    BODBA expects:
      - img: original image, shape (1, H, W, C)
      - q: query counter
      - decision(candidate): returns True if candidate is adversarial
    """

    def __init__(self):
        self.img = np.zeros((1, 32, 32, 3), dtype=np.float32)
        self.q = 0
        self.threshold = 0.04

    def decision(self, candidate):
        self.q += 1

        candidate = np.asarray(candidate, dtype=np.float32)

        assert candidate.shape == self.img.shape, (
            f"Bad candidate shape: {candidate.shape}; expected {self.img.shape}"
        )
        assert np.isfinite(candidate).all(), "Candidate contains NaN or inf"
        assert candidate.min() >= 0.0, f"Candidate min is {candidate.min()}"
        assert candidate.max() <= 1.0, f"Candidate max is {candidate.max()}"

        # Artificial adversarial condition:
        # any image bright enough is considered adversarial.
        return float(candidate.mean()) > self.threshold


def check_distortion_builders():
    img = DummyImage()

    cases = [
        ("perlin", np.array([50.0, 2.0, 8.0], dtype=np.float32)),
        ("gabor", np.array([2.0, 7.0, 3.0, 1.0, 4.0, 3.0], dtype=np.float32)),
        ("BICU", np.zeros(48, dtype=np.float32)),
        ("BILI", np.zeros(48, dtype=np.float32)),
        ("NN", np.zeros(48, dtype=np.float32)),
        ("CLUSTER", np.zeros(48, dtype=np.float32)),
    ]

    for noise, params in cases:
        out = create_distorted_image(
            image=img.img,
            typ=noise,
            epsilon=1.0,
            parameters=params,
        )
        assert out.shape == img.img.shape, f"{noise}: bad shape {out.shape}"
        assert np.isfinite(out).all(), f"{noise}: NaN or inf"
        assert out.min() >= 0.0 and out.max() <= 1.0, (
            f"{noise}: output outside [0, 1]"
        )

    print("distortion builders OK")


def check_quick_attack():
    img = DummyImage()

    history, adv = bayesian_attack(
        img,
        max_query=12,
        init_query=1,
        noise="perlin",
        max_norm=255,
        constraint="l2",
        seed=0,
        raw_samples=4,
        num_restarts=1,
        maxiter=2,
        show_progress=False,
    )

    assert isinstance(history, list), "history should be a list"
    assert len(history) >= 2, "history should contain multiple entries"
    assert img.q <= 12, f"query budget exceeded: q={img.q}"
    assert adv is not None, "expected to find a dummy adversarial image"
    assert adv.shape == img.img.shape, f"bad adversarial shape: {adv.shape}"
    assert np.isfinite(adv).all(), "adversarial image contains NaN or inf"
    assert adv.min() >= 0.0 and adv.max() <= 1.0, "adversarial image outside [0, 1]"
    assert float(adv.mean()) > img.threshold, "returned image is not adversarial"

    print("quick attack OK")
    print("queries:", img.q)
    print("last history row:", history[-1])
    print("adv shape:", adv.shape)
    print("adv mean:", float(adv.mean()))


def check_bo_path():
    """
    This uses at least two observations so the BoTorch GP proposal path is used.
    """
    img = DummyImage()

    history, adv = bayesian_attack(
        img,
        max_query=28,
        init_query=2,
        noise="perlin",
        max_norm=255,
        constraint="l2",
        seed=123,
        raw_samples=8,
        num_restarts=1,
        maxiter=5,
        show_progress=False,
    )

    assert isinstance(history, list)
    assert img.q <= 28, f"query budget exceeded: q={img.q}"
    assert adv is not None, "expected to find a dummy adversarial image"
    assert adv.shape == img.img.shape

    print("BoTorch GP path OK")
    print("queries:", img.q)
    print("last history row:", history[-1])


if __name__ == "__main__":
    check_distortion_builders()
    check_quick_attack()

    if "--bo" in sys.argv:
        check_bo_path()

    print("modernized BODBA smoke test passed")
