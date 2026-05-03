"""
inference.py — TFLite-based fast inference for skin disease classification.

Supports both deployment targets:
  • Raspberry Pi 4  → tflite-runtime  (lightweight)
  • Mac / dev       → tensorflow.lite  (bundled with TF)

The interpreter is loaded once at startup and reused for every prediction.
"""
# Postponed annotation evaluation — makes tuple[...] / X | Y work on Python 3.9.
from __future__ import annotations

import logging
import numpy as np

import config

log = logging.getLogger(__name__)

# Prefer the lightweight standalone runtime; fall back to full TensorFlow.
try:
    import tflite_runtime.interpreter as _tflite
    _Interpreter = _tflite.Interpreter
except ImportError:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter


CLASSES = config.CLASSES


class TFLiteClassifier:
    """Wraps a TFLite model for single-image classification."""

    def __init__(self, model_path: str):
        self._interpreter = _Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()

        self._in_idx  = self._interpreter.get_input_details()[0]["index"]
        self._out_idx = self._interpreter.get_output_details()[0]["index"]

        # Warm up: allocate buffers by running a blank pass.
        dummy = np.zeros(
            self._interpreter.get_input_details()[0]["shape"], dtype=np.float32
        )
        self._interpreter.set_tensor(self._in_idx, dummy)
        self._interpreter.invoke()

    # ------------------------------------------------------------------
    def predict(self, input_array: np.ndarray) -> tuple[str, float, int]:
        """
        Run inference on a preprocessed image array.

        Args:
            input_array : np.ndarray  shape (1, 224, 224, 3)  float32

        Returns:
            class_name  : str    e.g. "melanoma"
            confidence  : float  e.g. 92.4   (percentage, 0-100)
            class_idx   : int    index into CLASSES
        """
        arr = input_array.astype(np.float32)
        log.debug("input  stats: min=%.4f  max=%.4f  mean=%.4f", arr.min(), arr.max(), arr.mean())

        self._interpreter.set_tensor(self._in_idx, arr)
        self._interpreter.invoke()

        logits = self._interpreter.get_tensor(self._out_idx)[0].astype(np.float32)
        log.debug("raw output: %s", np.array2string(logits, precision=4, suppress_small=True))

        # Apply softmax only when output is raw logits (not already probabilities).
        probs = _softmax(logits) if not _is_distribution(logits) else logits
        log.debug("probs:  %s", dict(zip(CLASSES, [f"{p*100:.1f}%" for p in probs])))

        class_idx  = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100.0
        class_name = CLASSES[class_idx]

        log.info("→ %s  (%.1f%%)", class_name, confidence)
        return class_name, confidence, class_idx


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _is_distribution(x: np.ndarray, tol: float = 0.02) -> bool:
    """Return True if x already looks like a probability distribution (sums ≈ 1)."""
    return bool(x.min() >= 0.0 and abs(x.sum() - 1.0) < tol)
