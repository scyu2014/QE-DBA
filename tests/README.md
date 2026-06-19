# Testing the Modernized BO-DBA Attack

The original repo included outdated libs and interfaces. This version modernizes QE-DBA implementation (mainly the BODBA.py file and its dependency files). Other attacks (Bayes, HJSA, RayS, SignOPT) have not been modernized yet as of June 19, 2026.  

This guide explains how to download the modernized QE-DBA repository, set up a clean Python environment, and run the BO-DBA tests step by step.

It is written for students who may be new to Git, Python virtual environments, VS Code, and Bayesian Optimization attack code.

---

## 0. What you are testing

The modernized BO-DBA code lives mainly in:

```text
python_files/BODBA.py
python_files/Upsample.py
python_files/Util.py
python_files/Bayes_util.py
```

The test files live in:

```text
tests/
├── README.md
├── smoke_bodba_no_model.py
├── integration_bodba_single_image.py
├── integration_bodba_single_image_matrix.py
└── integration_bodba_real_model.py          # optional; requires an ImageNet-style dataset
```

The most important tests are:

| Test file | Purpose | Needs real image? | Needs model? |
|---|---|---:|---:|
| `tests/smoke_bodba_no_model.py` | Checks the BO attack loop with a fake image object | No | No |
| `tests/integration_bodba_single_image.py` | Runs the attack on one local `.jpg` or `.png` image | Yes | Yes |
| `tests/integration_bodba_single_image_matrix.py` | Runs several perturbation modes on one image | Yes | Yes |
| `tests/integration_bodba_real_model.py` | Samples from an ImageNet-style folder | Yes | Yes |

For a first test, run the files in this order:

```text
1. smoke_bodba_no_model.py
2. integration_bodba_single_image.py
3. integration_bodba_single_image_matrix.py
```

Do **not** start with `integration_bodba_real_model.py` unless you already have an ImageNet-style dataset folder.

---

## 1. Vocabulary

A few terms used in this guide:

| Term | Meaning |
|---|---|
| Repository, or repo | The project folder downloaded from GitHub |
| Clone | Download a GitHub repo to your computer using Git |
| Branch | A named version of the repo, for example `QE_DBA_New` |
| Terminal | The command-line panel in VS Code or your computer |
| Virtual environment | A private Python installation for this project only |
| Dependency | A Python package needed by the project, such as `torch`, `botorch`, or `tensorflow` |
| BO | Bayesian Optimization |
| BO-DBA attack | The Bayesian Optimization decision-based attack implemented in `BODBA.py` |
| Adversarial image | A modified image that causes the classifier to change its prediction |
| Query | One call to the model's decision function |

---

## 2. Install the basic tools

You need three tools:

```text
Git
Python 3.11
VS Code
```

Python 3.11 is recommended for this modernized version.

### macOS

If you have Homebrew:

```bash
brew update
brew install git
brew install python@3.11
```

Check that Git works:

```bash
git --version
```

Check that Python 3.11 works:

```bash
python3.11 --version
```

If `python3.11` is not found, try:

```bash
$(brew --prefix python@3.11)/bin/python3.11 --version
```

If that works, use the longer `$(brew --prefix python@3.11)/bin/python3.11` command whenever this guide says `python3.11`.

### Windows

Install:

```text
Git for Windows
Python 3.11 from python.org
VS Code
```

After installing Python, open PowerShell and check:

```powershell
py -3.11 --version
```

If this fails, reinstall Python 3.11 and make sure the Python launcher is enabled.

### Linux / Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y git python3.11 python3.11-venv python3.11-dev
```

Check:

```bash
git --version
python3.11 --version
```

---

## 3. Download the repo from GitHub

Open a terminal and choose a location for your project. For example, on macOS or Linux:

```bash
cd ~/Documents/Projects
```

If that folder does not exist:

```bash
mkdir -p ~/Documents/Projects
cd ~/Documents/Projects
```

Clone the modernized branch:

```bash
git clone -b QE_DBA_New https://github.com/scyu2014/QE-DBA.git
cd QE-DBA
```

If your instructor gives you a different GitHub URL or branch name, replace the URL and branch name above.

If the branch has already been merged into `main`, use:

```bash
git clone https://github.com/scyu2014/QE-DBA.git
cd QE-DBA
```

Check which branch you are on:

```bash
git branch --show-current
```

You should see either:

```text
QE_DBA_New
```

or:

```text
main
```

Open the project in VS Code:

```bash
code .
```

If `code .` does not work, open VS Code manually and use:

```text
File -> Open Folder -> QE-DBA
```

---

## 4. Check that the important files exist

From the VS Code terminal, run:

```bash
ls
```

You should see files and folders like:

```text
README.md
python_files
requirements-modern.txt
requirements-modern-minimal.txt
tests
```

Then check the tests folder:

```bash
ls tests
```

You should see:

```text
smoke_bodba_no_model.py
integration_bodba_single_image.py
integration_bodba_single_image_matrix.py
```

If those files are missing, you are probably on the wrong branch.

Check your branch again:

```bash
git branch --show-current
```

---

## 5. Create a Python virtual environment

A virtual environment keeps this project's Python packages separate from the rest of your computer.

Run the command from the repo root, meaning the folder that contains `python_files/` and `requirements-modern-minimal.txt`.

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If `python3.11` is not found on macOS Homebrew, use:

```bash
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Confirm that the virtual environment is active

macOS / Linux:

```bash
which python
python --version
```

Windows:

```powershell
where python
python --version
```

You should see a Python path inside the project folder, such as:

```text
.../QE-DBA/.venv/bin/python
```

or on Windows:

```text
...\QE-DBA\.venv\Scripts\python.exe
```

---

## 6. Install the Python dependencies

First upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

For testing the BO-DBA attack path, install the minimal modern requirements:

```bash
python -m pip install -r requirements-modern-minimal.txt
```

This installs packages such as:

```text
numpy
scipy
opencv-python-headless
tensorflow
torch
botorch
gpytorch
```

This step can take several minutes because TensorFlow, PyTorch, and BoTorch are large packages.

If you also want notebooks, install the full requirements instead:

```bash
python -m pip install -r requirements-modern.txt
```

You usually do **not** need both. For tests, `requirements-modern-minimal.txt` is enough.

---

## 7. Tell VS Code to use the project environment

In VS Code:

```text
Cmd/Ctrl + Shift + P
```

Search for:

```text
Python: Select Interpreter
```

Choose the interpreter inside `.venv`.

It should look similar to:

```text
.venv/bin/python
```

or on Windows:

```text
.venv\Scripts\python.exe
```

If you do not see `Python: Select Interpreter`, install the VS Code extension named:

```text
Python by Microsoft
```

Then reload VS Code:

```text
Cmd/Ctrl + Shift + P -> Developer: Reload Window
```

---

## 8. Run a basic import and syntax check

From the repo root:

```bash
python -m py_compile \
  python_files/BODBA.py \
  python_files/Upsample.py \
  python_files/Util.py \
  python_files/Bayes_util.py
```

No output means this passed.

Then run:

```bash
python -c "from python_files.BODBA import bayesian_attack; print('BODBA import OK')"
```

Expected output:

```text
BODBA import OK
```

---

## 9. Run the no-model smoke test

This test does not need a real image or a real classifier. It uses a fake image object to make sure the BO attack loop runs.

Run:

```bash
python tests/smoke_bodba_no_model.py
```

Expected output should include:

```text
distortion builders OK
quick attack OK
modernized BODBA smoke test passed
```

Now run the version that exercises the BoTorch Gaussian Process path:

```bash
python tests/smoke_bodba_no_model.py --bo
```

Expected output should include:

```text
BoTorch GP path OK
modernized BODBA smoke test passed
```

If this test passes, the modernized BO code is structurally working.

---

## 10. Prepare one image for the real-model test

Find any local image file on your computer, for example:

```text
kitten.jpg
dog.png
car.jpeg
```

The image can be anywhere, but it is easiest to put it on your Desktop.

Example path on macOS:

```text
~/Desktop/test.jpg
```

Example path on Windows:

```text
C:\Users\YourName\Desktop\test.jpg
```

Do **not** commit this image to Git unless your instructor specifically asks for it.

---

## 11. Run the single-image real-model test

This test loads one image, gets the model's original prediction, and runs the BO attack against that image.

macOS / Linux example:

```bash
python tests/integration_bodba_single_image.py ~/Desktop/test.jpg
```

Windows PowerShell example:

```powershell
python tests/integration_bodba_single_image.py "C:\Users\YourName\Desktop\test.jpg"
```

If your path contains spaces, use quotes:

```bash
python tests/integration_bodba_single_image.py "/Users/yourname/Desktop/my test image.jpg"
```

The first run may take longer because TensorFlow may download pretrained model weights.

Expected output should look roughly like:

```text
Initial prediction: ...
Initial query count: 0
queries: ...
history tail: ...
found adversary: True
```

or:

```text
found adversary: False
```

`found adversary: False` does **not** automatically mean the code failed. With a small query budget, the attack may simply not find an adversarial example yet.

The test is successful if:

```text
There is no Python traceback
The script reaches the final print statements
The query count is shown
```

---

## 12. Run the single-image matrix test

This test runs several perturbation modes:

```text
perlin
gabor
BICU
BILI
NN
CLUSTER
```

Run:

```bash
python tests/integration_bodba_single_image_matrix.py ~/Desktop/test.jpg
```

Expected output should contain sections like:

```text
================================================================================
Testing: perlin
status: OK
...
================================================================================
Testing: gabor
status: OK
...
```

It is okay if some modes do not find an adversarial image within 30 queries. The main goal of this test is to check that each mode runs without crashing.

---

## 13. Run a larger attack after the small tests pass

The integration tests use small query budgets so they finish quickly. To give the BO attack a better chance of success, increase the budget.

Open:

```text
tests/integration_bodba_single_image.py
```

Find the call to `bayesian_attack(...)` and change it to something like:

```python
history, adv = bayesian_attack(
    img,
    max_query=200,
    init_query=3,
    noise="perlin",
    max_norm=255,
    constraint="l2",
    seed=0,
    raw_samples=16,
    num_restarts=3,
    maxiter=20,
    show_progress=True,
)
```

Then rerun:

```bash
python tests/integration_bodba_single_image.py ~/Desktop/test.jpg
```

A larger query budget gives the Bayesian Optimization loop more chances to find an adversarial example.

---

## 14. Optional: test with an ImageNet-style dataset folder

Only use this test if you have a folder like this:

```text
QE-DBA/
└── DataSet/
    └── ILSVRC2012_img_val/
        ├── n02123045/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── n02084071/
        │   └── image3.jpg
        └── ...
```

The folder names should be ImageNet class IDs such as:

```text
n02123045
n02084071
n01440764
```

Then run:

```bash
python tests/integration_bodba_real_model.py
```

If you see this error:

```text
FileNotFoundError: No ImageNet-style class directories found at './DataSet/ILSVRC2012_img_val'
```

it means you do not have the expected ImageNet-style dataset folder. Use the single-image test instead:

```bash
python tests/integration_bodba_single_image.py ~/Desktop/test.jpg
```

---

## 15. How to interpret the results

### Case 1: smoke test passes

If this passes:

```bash
python tests/smoke_bodba_no_model.py --bo
```

then the BoTorch-based BO loop is working without a real classifier.

### Case 2: single-image test passes but no adversary is found

This is usually fine.

The attack may need more queries, a different image, or a different perturbation mode.

Try:

```text
Increase max_query
Try noise="gabor"
Try a different image
Use the matrix test
```

### Case 3: matrix test passes

If every section says:

```text
status: OK
```

then all perturbation modes are at least runnable.

### Case 4: ImageNet dataset test fails

This usually means the dataset folder is missing or has the wrong structure. It does not mean the BO attack code is broken.

---

## 16. Common errors and fixes

### Error: `zsh: command not found: python3.11`

Python 3.11 is not installed or not on your shell path.

On macOS with Homebrew:

```bash
brew install python@3.11
$(brew --prefix python@3.11)/bin/python3.11 --version
```

Then create the environment with:

```bash
$(brew --prefix python@3.11)/bin/python3.11 -m venv .venv
source .venv/bin/activate
```

---

### Error: `ModuleNotFoundError: No module named 'botorch'`

The requirements were not installed in the active environment.

Run:

```bash
source .venv/bin/activate
python -m pip install -r requirements-modern-minimal.txt
```

Then test:

```bash
python -c "import botorch; print('botorch OK')"
```

---

### Error: `ModuleNotFoundError: No module named 'python_files'`

You are probably running the test from the wrong folder.

Go to the repo root:

```bash
cd /path/to/QE-DBA
```

Then run:

```bash
python tests/smoke_bodba_no_model.py
```

Do not run the test from inside the `tests/` folder unless you know how to set `PYTHONPATH`.

---

### Error: `FileNotFoundError: No ImageNet-style class directories found`

You ran:

```bash
python tests/integration_bodba_real_model.py
```

but you do not have the ImageNet-style dataset folder.

Use this instead:

```bash
python tests/integration_bodba_single_image.py ~/Desktop/test.jpg
```

---

### Message: `Matplotlib is building the font cache; this may take a moment.`

This is normal. It usually happens the first time Matplotlib runs.

Wait for it to finish.

---

### Error during TensorFlow model loading

The first real-model test may download pretrained model weights. Make sure you have internet access.

Try rerunning:

```bash
python tests/integration_bodba_single_image.py ~/Desktop/test.jpg
```

If the error continues, copy the full traceback and check:

```bash
python --version
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

### Error: branch push or branch name problem

Check your current branch:

```bash
git branch --show-current
```

Push the current branch safely with:

```bash
git push -u origin HEAD
```

This avoids typing the branch name incorrectly.

---

## 17. Files that should not be committed

Do not commit local environment files, cache files, or personal test images.

These should usually be ignored:

```text
.DS_Store
.venv/
DataSet/
_patch/
__pycache__/
*.pyc
```

A useful `.gitignore` section is:

```gitignore
# macOS
.DS_Store

# Python cache files
__pycache__/
*.py[cod]
.pytest_cache/

# Local Python environments
.venv/
venv/
env/

# Local datasets and personal test images
DataSet/

# Temporary patch folder
_patch/
```

---

## 18. Quick command summary

For macOS / Linux, the shortest successful path is:

```bash
git clone -b QE_DBA_New https://github.com/scyu2014/QE-DBA.git
cd QE-DBA
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-modern-minimal.txt
python -m py_compile python_files/BODBA.py python_files/Upsample.py python_files/Util.py python_files/Bayes_util.py
python tests/smoke_bodba_no_model.py --bo
python tests/integration_bodba_single_image.py ~/Desktop/test.jpg
python tests/integration_bodba_single_image_matrix.py ~/Desktop/test.jpg
```

For Windows PowerShell:

```powershell
git clone -b QE_DBA_New https://github.com/scyu2014/QE-DBA.git
cd QE-DBA
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-modern-minimal.txt
python -m py_compile python_files/BODBA.py python_files/Upsample.py python_files/Util.py python_files/Bayes_util.py
python tests/smoke_bodba_no_model.py --bo
python tests/integration_bodba_single_image.py "C:\Users\YourName\Desktop\test.jpg"
python tests/integration_bodba_single_image_matrix.py "C:\Users\YourName\Desktop\test.jpg"
```

---

## 19. What to report if something fails

When asking for help, include:

```text
1. Your operating system: macOS, Windows, or Linux
2. The command you ran
3. The full error message / traceback
4. Output of: python --version
5. Output of: git branch --show-current
6. Which test file failed
```

Useful commands:

```bash
python --version
git branch --show-current
git status
python -c "import torch, botorch, gpytorch, tensorflow as tf; print('imports OK')"
```

