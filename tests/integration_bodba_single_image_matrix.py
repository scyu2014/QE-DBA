from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_files.Util import importimage, randomimg
from python_files.BODBA import bayesian_attack


if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python tests/integration_bodba_single_image_matrix.py /path/to/image.jpg"
    )

image_path = sys.argv[1]

_, base_image = importimage(image_path)

cases = [
    {
        "noise": "perlin",
        "max_query": 30,
        "init_query": 1,
        "raw_samples": 4,
        "num_restarts": 1,
        "maxiter": 2,
    },
    {
        "noise": "gabor",
        "max_query": 30,
        "init_query": 1,
        "raw_samples": 4,
        "num_restarts": 1,
        "maxiter": 2,
    },
    {
        "noise": "BICU",
        "max_query": 30,
        "init_query": 2,
        "raw_samples": 4,
        "num_restarts": 1,
        "maxiter": 2,
    },
    {
        "noise": "BILI",
        "max_query": 30,
        "init_query": 2,
        "raw_samples": 4,
        "num_restarts": 1,
        "maxiter": 2,
    },
    {
        "noise": "NN",
        "max_query": 30,
        "init_query": 2,
        "raw_samples": 4,
        "num_restarts": 1,
        "maxiter": 2,
    },
    {
        "noise": "CLUSTER",
        "max_query": 30,
        "init_query": 2,
        "raw_samples": 4,
        "num_restarts": 1,
        "maxiter": 2,
    },
]

for case in cases:
    print("=" * 80)
    print("Testing:", case["noise"])

    img = randomimg(img=base_image)

    try:
        history, adv = bayesian_attack(
            img,
            noise=case["noise"],
            max_query=case["max_query"],
            init_query=case["init_query"],
            max_norm=255,
            constraint="l2",
            seed=0,
            raw_samples=case["raw_samples"],
            num_restarts=case["num_restarts"],
            maxiter=case["maxiter"],
            show_progress=False,
        )

        print("status: OK")
        print("queries:", img.q)
        print("history length:", len(history))
        print("found adversary:", adv is not None)

        if history:
            print("last history row:", history[-1])

        if adv is not None:
            print("adv shape:", adv.shape)
            print("adv min/max:", float(adv.min()), float(adv.max()))

    except Exception:
        print("status: FAILED")
        traceback.print_exc()
