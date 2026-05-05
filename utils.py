import numpy as np
from PIL import Image
import config

NORM_MODE = config.NORM_MODE

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = (224, 224)

def preprocess_image(source, target_size: tuple = INPUT_SIZE):

    if isinstance(source, str):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source).convert("RGB")
    else:
        img = source.convert("RGB")

    original_img = img.copy()
    img_resized  = img.resize(target_size, Image.LANCZOS)
    arr          = np.array(img_resized, dtype=np.float32)

    if NORM_MODE == "simple":
        arr = arr / 255.0
    elif NORM_MODE == "raw":
        pass
    elif NORM_MODE == "tf":
        arr = arr / 127.5 - 1.0
    elif NORM_MODE == "imagenet":
        arr = arr / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    else:
        raise ValueError(f"Unknown NORM_MODE: {NORM_MODE!r}")

    return np.expand_dims(arr, axis=0), original_img

def resize_for_display(pil_image: Image.Image, max_size: tuple = (420, 320)) -> Image.Image:

    img = pil_image.copy()
    img.thumbnail(max_size, Image.LANCZOS)
    return img
