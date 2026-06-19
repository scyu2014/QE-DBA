from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_files.Util import importimage, randomimg
from python_files.BODBA import bayesian_attack


if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python tests/integration_bodba_single_image.py /path/to/image.jpg"
    )

image_path = sys.argv[1]

_, image = importimage(image_path)

# This bypasses DataSet/ILSVRC2012_img_val.
# It uses the model's prediction on this image as the original label.
img = randomimg(img=image)

print("Initial prediction:", img.image_probs[0])
print("Initial query count:", img.q)

history, adv = bayesian_attack(
    img,
    max_query=30,
    init_query=1,
    noise="perlin",
    max_norm=255,
    constraint="l2",
    seed=0,
    raw_samples=4,
    num_restarts=1,
    maxiter=2,
    show_progress=True,
)

print("queries:", img.q)
print("history tail:", history[-5:])
print("found adversary:", adv is not None)

if adv is not None:
    print("adversary shape:", adv.shape)
    print("adversary min/max:", float(adv.min()), float(adv.max()))
