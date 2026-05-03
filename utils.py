"""
utils.py — Image loading and preprocessing utilities.

If all predictions collapse to a single class, the normalization is wrong.
Change NORM_MODE below to match whatever was used during training:

  "simple"   →  x / 255.0              range [0, 1]   ← try first
  "raw"      →  x as float32           range [0, 255] (model has built-in rescaling)
  "tf"       →  x / 127.5 - 1.0        range [-1, 1]  (legacy EfficientNet tf-mode)
  "imagenet" →  subtract ImageNet mean/std              (PyTorch / torch-mode)
"""

import numpy as np
from PIL import Image
import config

# NORM_MODE is read from config.py — edit that file to change it.
NORM_MODE = config.NORM_MODE

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = (224, 224)


def preprocess_image(source, target_size: tuple = INPUT_SIZE):
    """
    Resize and normalise an image for model inference.

    Args:
        source     : str (file path), np.ndarray, or PIL.Image
        target_size: (width, height), default (224, 224)

    Returns:
        input_array  : np.ndarray  shape (1, H, W, 3)  float32
        original_img : PIL.Image   at original resolution
    """
    if isinstance(source, str):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source).convert("RGB")
    else:
        img = source.convert("RGB")

    original_img = img.copy()
    img_resized  = img.resize(target_size, Image.LANCZOS)
    arr          = np.array(img_resized, dtype=np.float32)   # [0, 255]

    if NORM_MODE == "simple":
        arr = arr / 255.0
    elif NORM_MODE == "raw":
        pass                                                  # keep [0, 255]
    elif NORM_MODE == "tf":
        arr = arr / 127.5 - 1.0
    elif NORM_MODE == "imagenet":
        arr = arr / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    else:
        raise ValueError(f"Unknown NORM_MODE: {NORM_MODE!r}")

    return np.expand_dims(arr, axis=0), original_img


def resize_for_display(pil_image: Image.Image, max_size: tuple = (420, 320)) -> Image.Image:
    """Return a copy of pil_image scaled to fit within max_size (aspect-preserving)."""
    img = pil_image.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    return img
