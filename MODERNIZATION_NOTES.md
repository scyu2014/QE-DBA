# QE-DBA modernization patch

This patch modernizes the BO-DBA path while preserving the public notebook-facing function:

```python
from python_files.BODBA import bayesian_attack
history, adversary = bayesian_attack(image, max_query=500, noise="perlin")
```

## Files included

- `python_files/BODBA.py`
  - Replaces `GPyOpt` with BoTorch/GPyTorch.
  - Uses `SingleTaskGP`, `fit_gpytorch_mll`, `LogExpectedImprovement`, and `optimize_acqf`.
  - Keeps `bayesian_attack(...)`, `preddifference`, `binary_search(...)`, and the old perturbation names.
  - Supports `noise="perlin"`, `"gabor"`, `"BICU"`, `"BILI"`, `"NN"`, and `"CLUSTER"`.

- `python_files/Upsample.py`
  - Rewrites up/down-sampling with consistent channel-first flattening.
  - Removes old `.data.numpy()` usage.
  - Vectorizes the `CLUSTER` path.

- `python_files/Util.py`
  - Removes standalone `keras` import.
  - Adds lazy classifier loading.
  - Fixes grayscale/RGBA image loading by converting to RGB.
  - Fixes MNIST/CIFAR label handling.
  - Keeps public functions used by the original notebook: `randomimg`, `norm`, `display_images`, `importimage`, `adversarial_detection`, `ResultSave`, and `DemoVisulization`.

- `python_files/Bayes_util.py`
  - Replaces removed `torch.irfft` calls with `torch.fft.ifft2`.
  - Keeps the original helper function names.

- `requirements-modern.txt`
  - Full modern environment for notebooks and the modern BO-DBA path.

- `requirements-modern-minimal.txt`
  - Smaller runtime dependency file for the modern `BODBA.py` path.

## Apply the patch

From the root of your forked repo:

```bash
unzip qe_dba_modernized_patch.zip -d /tmp/qe_dba_patch
cp -R /tmp/qe_dba_patch/python_files ./
cp /tmp/qe_dba_patch/requirements-modern*.txt ./
```

Then create a new environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-modern.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-modern.txt
```

## Important behavior changes

1. `GPyOpt` is removed. The Bayesian optimization loop now uses BoTorch.
2. The modern default target is Python 3.11/3.12, not Python 3.8.
3. The Perlin perturbation no longer requires the third-party `noise` C extension. It uses a deterministic numpy-based fBm-style substitute with the same three-parameter interface.
4. `Upsample.py` now returns channel-first flattened arrays consistently. This fixes the old mixed-layout behavior between `CLUSTER` and interpolation modes.
5. `Util.py` loads the configured classifier lazily. Importing the module should no longer immediately download ImageNet weights.
6. The code has been syntax-checked here, but it has not been end-to-end validated against the ImageNet dataset or the repo's original notebooks because those assets are not present in this environment.

## Suggested smoke test

```python
from python_files.Util import randomimg
from python_files.BODBA import bayesian_attack

img = randomimg()
history, adv = bayesian_attack(
    img,
    max_query=100,
    init_query=5,
    noise="perlin",
    max_norm=16,
    constraint="l2",
    seed=0,
)
print(history[-1])
print(adv is not None)
```
