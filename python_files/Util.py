"""Modern utility layer for QE-DBA.

This file preserves the public names used by the notebooks and attack modules
while removing several legacy failure modes:

* no standalone ``keras`` import; use ``tf.keras``;
* lazy model/dataset loading instead of heavy import-time side effects;
* safe RGB conversion for grayscale/RGBA inputs;
* correct MNIST/CIFAR label handling;
* robust image/norm conversion for numpy, TensorFlow tensors, and torch tensors.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from scipy import ndimage


def SAVE(fp: str, input_obj) -> None:
    with open(fp, "wb+") as handle:
        pickle.dump(input_obj, handle)


def LOAD(fp: str):
    with open(fp, "rb+") as handle:
        return pickle.load(handle)


def millis() -> int:
    return int(round(time.time() * 1000))


def _read_config(config_path: str = "./Configuration.yaml") -> dict:
    defaults = {
        "main_path": "./",
        "data_path": "./DataSet/ILSVRC2012_img_val",
        "mod": "Inception",
    }
    path = Path(config_path)
    if not path.exists():
        return defaults
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return {**defaults, **loaded}


params = _read_config()
main_path = str(params["main_path"])
data_path = str(params["data_path"])
mod = str(params["mod"])


if mod == "ResNet":
    imagesize = 224
    decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
elif mod == "Inception":
    imagesize = 299
    decode_predictions = tf.keras.applications.inception_v3.decode_predictions
elif mod == "MNIST":
    imagesize = 28
    decode_predictions = None
elif mod == "CIFAR":
    imagesize = 32
    decode_predictions = None
else:
    raise ValueError("Configuration.yaml 'mod' must be ResNet, Inception, MNIST, or CIFAR.")


def _list_imagenet_files(root: str) -> Tuple[List[str], Dict[str, List[str]]]:
    root_path = Path(root)
    if not root_path.exists():
        return [], {}
    classes = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    files: Dict[str, List[str]] = {}
    for cls in classes:
        class_dir = root_path / cls
        files[cls] = sorted([p.name for p in class_dir.iterdir() if p.is_file()])
    return classes, files


classfiles, images = _list_imagenet_files(data_path)
cls = random.choice(classfiles) if classfiles else None


_MODEL = None
_DATASET_CACHE = {}


class LazyModelProxy:
    """Proxy that preserves ``pretrained_model.predict(...)`` compatibility."""

    def __getattr__(self, name):
        return getattr(get_model(), name)

    def predict(self, *args, **kwargs):
        return get_model().predict(*args, **kwargs)

    @property
    def trainable(self):
        return get_model().trainable

    @trainable.setter
    def trainable(self, value):
        get_model().trainable = value


pretrained_model = LazyModelProxy()


def _load_custom_model(path: str):
    """Load a local Keras model.

    Keras 3 changed some SavedModel behavior. For standard ``.keras``/``.h5``
    files and many TensorFlow SavedModels, ``tf.keras.models.load_model`` still
    works. If your old model directory fails to load, convert it to the modern
    ``.keras`` format once, then update ``Configuration.yaml``/paths.
    """
    return tf.keras.models.load_model(path)


def get_model():
    """Return the configured classifier, loading it on first use."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if mod == "ResNet":
        _MODEL = tf.keras.applications.ResNet50V2(weights="imagenet")
    elif mod == "Inception":
        _MODEL = tf.keras.applications.InceptionV3(weights="imagenet")
    elif mod == "MNIST":
        _MODEL = _load_custom_model("./Models/Mnist")
    elif mod == "CIFAR":
        _MODEL = _load_custom_model("./Models/CIFAR10")
    else:  # defensive; checked above.
        raise ValueError(f"Unsupported model type {mod!r}.")
    _MODEL.trainable = False
    return _MODEL


def _load_dataset(name: str):
    if name in _DATASET_CACHE:
        return _DATASET_CACHE[name]
    if name == "MNIST":
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = np.expand_dims(x_test, -1)
        y_test = y_test.astype("int64")
    elif name == "CIFAR":
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype("float32") / 255.0
        y_test = y_test.reshape(-1).astype("int64")
    else:
        raise ValueError(name)
    _DATASET_CACHE[name] = (x_test, y_test)
    return x_test, y_test


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        arr = value.numpy()
    else:
        arr = np.asarray(value)
    return arr.astype(np.float32, copy=False)


def _ensure_batch(value) -> np.ndarray:
    arr = _to_numpy(value)
    if arr.ndim == 2:
        arr = arr[None, :, :, None]
    elif arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected image shape (1, H, W, C); got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def norm(image, image2) -> Tuple[float, float]:
    y = _ensure_batch(image)
    z = _ensure_batch(image2)
    diff = (z - y).reshape(-1)
    l2norm = float(np.linalg.norm(diff, ord=2))
    infnorm = float(np.max(np.abs(diff)))
    return l2norm, infnorm


def _predict(image) -> np.ndarray:
    batch = _ensure_batch(image)
    return np.asarray(get_model().predict(batch, steps=1, verbose=0))


def get_imagenet_label(probs) -> List[Tuple[object, str, float]]:
    probs = np.asarray(probs)
    if mod in {"MNIST", "CIFAR"}:
        vector = probs[0]
        idxlist = np.argsort(-vector)
        return [(int(idx), str(int(idx)), float(vector[idx])) for idx in idxlist[:6]]
    if decode_predictions is None:
        raise RuntimeError("decode_predictions is not configured for this model.")
    decoded = decode_predictions(probs, top=6)[0]
    return [(item[0], item[1], float(item[2])) for item in decoded]


def display_images(image) -> None:
    batch = _ensure_batch(image)
    guessdata = get_imagenet_label(_predict(batch))
    for guess in guessdata:
        print(f"{guess[1]}: {guess[2]}")
    fig = plt.figure()
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.squeeze(batch[0]), cmap="gray" if batch.shape[-1] == 1 else None)
    plt.show()


def importimage(imgpath: str):
    """Load an image as a normalized RGB tensor with shape ``(1, H, W, 3)``."""
    rawimage = Image.open(imgpath)
    rgb = rawimage.convert("RGB")
    image = tf.keras.utils.img_to_array(rgb, dtype="float32") / 255.0
    image = tf.image.resize(image, (imagesize, imagesize)).numpy().astype(np.float32)
    image = image[None, ...]
    return rawimage, image


class randomimg:
    """Random correctly-classified image wrapper used by the attacks."""

    def __init__(
        self,
        m: str = "joint",
        t: float = -1,
        mode: str = "Raw",
        target: int = 0,
        img=None,
        label=None,
        max_attempts: int = 100,
    ):
        self.maxl2 = float("inf")
        self.maxlinf = float("inf")
        self.historyl2: List[List[float]] = []
        self.historylinf: List[List[float]] = []
        self.q = 0
        self.method = m
        self.threshold = t
        self.mode = mode
        self.target = target

        if img is not None:
            self.img = _ensure_batch(img)
            self.image_probs = get_imagenet_label(_predict(self.img))
            self.labelindex = self.image_probs[0][0]
            self.actualprediction = label
            return

        for _ in range(max_attempts):
            if mod == "MNIST":
                x_test, y_test = _load_dataset("MNIST")
                index = random.randrange(len(x_test))
                image = np.expand_dims(x_test[index], 0).astype(np.float32)
                actual = int(y_test[index])
            elif mod == "CIFAR":
                x_test, y_test = _load_dataset("CIFAR")
                index = random.randrange(len(x_test))
                image = np.expand_dims(x_test[index], 0).astype(np.float32)
                actual = int(y_test[index])
            else:
                if not classfiles:
                    raise FileNotFoundError(
                        f"No ImageNet-style class directories found at {data_path!r}. "
                        "Set data_path in Configuration.yaml."
                    )
                chosen_cls = random.choice(classfiles)
                if not images.get(chosen_cls):
                    continue
                imgfile = random.choice(images[chosen_cls])
                imgpath = os.path.join(data_path, chosen_cls, imgfile)
                _, image = importimage(imgpath)
                actual = chosen_cls

            probs = _predict(image)
            labels = get_imagenet_label(probs)
            predicted = labels[0][0]
            if predicted == actual:
                self.img = image
                self.image_probs = labels
                self.labelindex = predicted
                self.actualprediction = actual
                return

        raise RuntimeError(
            f"Could not sample a correctly-classified {mod} image within {max_attempts} attempts."
        )

    def update(self, img) -> None:
        twonorm, infnorm = norm(self.img, img)
        if twonorm < self.maxl2:
            self.maxl2 = twonorm
            self.historyl2.append([float(self.q), float(twonorm)])
        if infnorm < self.maxlinf:
            self.maxlinf = infnorm
            self.historylinf.append([float(self.q), float(infnorm)])

    def decision(self, img) -> bool:
        batch = _ensure_batch(img)
        check = get_imagenet_label(_predict(batch))
        self.q += 1

        if self.mode == "Target":
            result = check[0][0] == self.target
        else:
            result = check[0][0] != self.image_probs[0][0]

        if self.mode == "Detection":
            if self.threshold == -1:
                detected = adversarial_detection(batch, self.method)[0]
            else:
                detected = adversarial_detection(batch, self.method, self.threshold)[0]
            if detected:
                result = False

        if result:
            self.update(batch)
        return bool(result)


# Defense methods.
def l1_dist(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sum(np.abs(x1 - x2), axis=tuple(range(1, x1.ndim)))


def median_smoothing(x, width: int, height: int = -1):
    batch = _ensure_batch(x)
    if height == -1:
        height = width
    # NHWC layout: do not smooth across batch or channels.
    return ndimage.median_filter(batch, size=(1, int(width), int(height), 1), mode="reflect").astype(np.float32)


def bit_depth(x, npp: int):
    batch = _ensure_batch(x)
    npp_int = int(npp) - 1
    if npp_int <= 0:
        raise ValueError("npp must be > 1.")
    x_int = np.rint(batch * npp_int)
    return (x_int / npp_int).astype(np.float32)


def non_local_mean(x, a: float, b: int, c: int):
    batch = _ensure_batch(x)
    image = np.clip(batch[0] * 255.0, 0, 255).astype(np.uint8)
    if image.shape[-1] == 1:
        denoised = cv2.fastNlMeansDenoising(image[..., 0], None, float(a), int(b), int(c))[..., None]
    else:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, float(a), float(a), int(b), int(c))
    return np.expand_dims(denoised.astype(np.float32) / 255.0, 0)


def adversarial_detection(im, method: str, threshold: float = -1):
    if threshold == -1:
        if method == "bit_depth":
            threshold = 0.307
        elif method == "median_smoothing":
            threshold = 0.940
        elif method == "non_local_mean":
            threshold = 0.623
        elif method == "joint":
            threshold = 1.307
        else:
            raise ValueError("method must be bit_depth, median_smoothing, non_local_mean, or joint.")

    batch = _ensure_batch(im)
    originalpred = _predict(batch)

    if method == "joint":
        dist1 = adversarial_detection(batch, "bit_depth")[1]
        dist2 = adversarial_detection(batch, "median_smoothing")[1]
        dist3 = adversarial_detection(batch, "non_local_mean")[1]
        preddist = np.maximum(np.maximum(dist1, dist2), dist3)
    else:
        if method == "bit_depth":
            squeezed = bit_depth(batch, 32)
        elif method == "median_smoothing":
            squeezed = median_smoothing(batch, 2)
        elif method == "non_local_mean":
            squeezed = non_local_mean(batch, 11, 3, 4)
        else:
            raise ValueError("method must be bit_depth, median_smoothing, non_local_mean, or joint.")
        newpred = _predict(squeezed)
        preddist = l1_dist(np.array([originalpred]), np.array([newpred]))

    return bool(np.any(preddist > threshold)), preddist


def getthreshold(imglist: Sequence[np.ndarray], dettype: str, percentile: float):
    distlist = []
    for orig in imglist:
        origdetect = adversarial_detection(orig, dettype)
        distlist.append(float(np.ravel(origdetect[1])[0]))
    print(np.percentile(distlist, percentile))
    cutoff = int(np.round(len(imglist) * (percentile / 100.0)) - 1)
    distlist = np.sort(distlist)
    print(distlist)
    return float(distlist[cutoff])


def ResultSave(Name: str, Path: str):
    file_name = f"{Path}/{Name}.dat"
    dir_path = f"{Path}/{Name}"
    os.makedirs(dir_path, exist_ok=True)
    img_prefix = f"{dir_path}/{Name}_"
    return file_name, img_prefix


def DemoVisulization(oriImg, Adversary, History, queryBudgets: int, fontsize: int = 20, SavePath: Optional[str] = None):
    ori = _ensure_batch(oriImg)
    adv = _ensure_batch(Adversary)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    ax1.imshow(np.squeeze(ori[0]), cmap="gray" if ori.shape[-1] == 1 else None)
    ax1.set_title("Original Image", size=fontsize)
    ax1.set_axis_off()
    predict = ""
    for guess in get_imagenet_label(_predict(ori)):
        predict += f"{guess[1]}: {guess[2]:.3f}\n"
    ax1.text(1, -0.01, predict, ha="right", va="top", size=fontsize * 0.8, transform=ax1.transAxes)

    ax2.imshow(np.squeeze(adv[0]), cmap="gray" if adv.shape[-1] == 1 else None)
    ax2.set_title("Adversarial Example", size=fontsize)
    ax2.set_axis_off()
    predict = ""
    for guess in get_imagenet_label(_predict(adv)):
        predict += f"{guess[1]}: {guess[2]:.3f}\n"
    ax2.text(1, -0.01, predict, ha="right", va="top", size=fontsize * 0.8, transform=ax2.transAxes)

    perturbation = np.clip((adv[0] - ori[0]) * 255.0 + 127.5, 0, 255).astype("uint8")
    ax3.imshow(np.squeeze(perturbation), cmap="gray" if perturbation.shape[-1] == 1 else None)
    ax3.set_title("Perturbation", size=fontsize)
    ax3.set_axis_off()
    fig.tight_layout()
    if SavePath is not None:
        plt.savefig(SavePath, bbox_inches="tight", pad_inches=0)

    x = range(queryBudgets)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["$l_2$ Distance", "$l_\\infty$ Distance", "Time(s)"]
    for ax, history, label in zip((ax1, ax2, ax3), History, labels):
        X1 = [point[0] for point in history]
        Y1 = [point[1] for point in history]
        if label == "Time(s)":
            Y1 = [v / 1000.0 for v in Y1]
        if len(X1) >= 2:
            Y1_interp = np.interp(list(x), X1, Y1)
            ax.plot(list(x), Y1_interp, "-")
        ax.set_title(label, size=fontsize)
    fig.tight_layout()
    return fig
