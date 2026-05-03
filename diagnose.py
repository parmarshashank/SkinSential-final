"""
diagnose.py — Run this ONCE to figure out the correct preprocessing.

Usage:
    python diagnose.py                         # uses a synthetic test image
    python diagnose.py path/to/skin_image.jpg  # uses a real image (recommended)

Prints:
  • TFLite model input/output metadata
  • Raw logits + probabilities for every normalization mode
  • Which mode gives the most spread-out (confident + varied) predictions
"""

import sys
import numpy as np
from PIL import Image

TFLITE_MODEL = "model_fp16.tflite"
CLASSES      = ["melanoma", "psoriasis", "ringworm", "normal"]
INPUT_SIZE   = (224, 224)

# ── Load image ────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    img = Image.open(sys.argv[1]).convert("RGB")
    print(f"Using image: {sys.argv[1]}")
else:
    # Synthetic skin-tone patch — better than all-black or all-white
    rng = np.random.default_rng(42)
    fake = rng.integers(100, 220, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(fake)
    print("No image provided — using a random synthetic patch.")
    print("TIP: run again with a real skin photo for meaningful results:\n"
          "  python diagnose.py /path/to/image.jpg\n")

arr_uint8 = np.array(img.resize(INPUT_SIZE, Image.LANCZOS), dtype=np.uint8)

# ── Load TFLite interpreter ───────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("Runtime: tflite_runtime")
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    print("Runtime: tensorflow.lite")

interp = Interpreter(model_path=TFLITE_MODEL)
interp.allocate_tensors()

in_detail  = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print("\n" + "="*60)
print("TFLite model — input tensor details")
print("="*60)
print(f"  name      : {in_detail['name']}")
print(f"  shape     : {in_detail['shape']}")
print(f"  dtype     : {in_detail['dtype']}")
quant = in_detail.get("quantization", (0.0, 0))
print(f"  quant     : scale={quant[0]:.6f}, zero_point={quant[1]}")
print(f"  quant_para: {in_detail.get('quantization_parameters', {})}")

print("\nTFLite model — output tensor details")
print(f"  name      : {out_detail['name']}")
print(f"  shape     : {out_detail['shape']}")
print(f"  dtype     : {out_detail['dtype']}")
quant_out = out_detail.get("quantization", (0.0, 0))
print(f"  quant     : scale={quant_out[0]:.6f}, zero_point={quant_out[1]}")

# ── Normalization modes to test ───────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def apply_norm(arr_u8, mode):
    a = arr_u8.astype(np.float32)
    if mode == "raw":
        return a
    if mode == "simple":
        return a / 255.0
    if mode == "tf":
        return a / 127.5 - 1.0
    if mode == "imagenet":
        return (a / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    raise ValueError(mode)

def run(arr_f32):
    inp = np.expand_dims(arr_f32, 0).astype(np.float32)
    interp.set_tensor(in_detail["index"], inp)
    interp.invoke()
    out = interp.get_tensor(out_detail["index"])[0].astype(np.float32)
    return out

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

# ── Run every mode and print results ─────────────────────────────────
print("\n" + "="*60)
print("Results for each normalization mode")
print("="*60)

best_mode    = None
best_entropy = -1.0

for mode in ["raw", "simple", "tf", "imagenet"]:
    arr = apply_norm(arr_uint8, mode)

    print(f"\n── NORM_MODE = {mode!r:10s}  "
          f"input range [{arr.min():.3f}, {arr.max():.3f}]  "
          f"mean={arr.mean():.3f}")

    raw_out = run(arr)
    print(f"   raw output : {np.array2string(raw_out, precision=4, suppress_small=True)}")

    # Decide whether to apply softmax
    if raw_out.min() >= -0.01 and abs(raw_out.sum() - 1.0) < 0.05:
        probs = raw_out
        note  = "(already probabilities)"
    else:
        probs = softmax(raw_out)
        note  = "(softmax applied)"

    print(f"   probs {note}:")
    for i, (cls, p) in enumerate(zip(CLASSES, probs)):
        bar = "█" * int(p * 30)
        print(f"     [{i}] {cls:<12s}  {p*100:6.2f}%  {bar}")

    predicted = CLASSES[int(np.argmax(probs))]
    conf      = float(probs.max()) * 100
    print(f"   → Prediction: {predicted.upper()}  ({conf:.1f}%)")

    # Entropy: higher = more spread = more likely correct preprocessing
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    print(f"   entropy (higher = better spread): {entropy:.4f}")

    if entropy > best_entropy:
        best_entropy = entropy
        best_mode    = mode

print("\n" + "="*60)
print(f"RECOMMENDATION:  NORM_MODE = {best_mode!r}")
print(f"(highest entropy = least collapsed predictions)")
print("="*60)
print(f"\nSet  NORM_MODE = {best_mode!r}  in utils.py and re-run the app.")
