# Postponed annotation evaluation — makes X | None / tuple[...] work on Python 3.9.
from __future__ import annotations

"""
gradcam.py — Gradient saliency map using the TensorFlow SavedModel.

Keras 3 (shipped with TF ≥ 2.16) dropped support for loading legacy SavedModel
format via keras.models.load_model().  We use tf.saved_model.load() instead,
which works with any TF version.

Because tf.saved_model.load() returns a concrete inference function (not a
layer-addressable Keras model), we compute a gradient saliency map rather than
true Grad-CAM.  The result is visually equivalent: pixels that most influenced
the prediction are highlighted in red/yellow.

Method: Gradient × Input saliency
  1. Forward pass through the serving_default signature.
  2. Compute ∂(class score) / ∂(input image) via GradientTape.
  3. |gradient| × |input|  →  per-pixel importance.
  4. Collapse RGB channels by mean, normalise to [0,1], overlay with JET colormap.
"""

import threading
import numpy as np
import cv2
from PIL import Image


class GradCAMExplainer:
    """Gradient saliency explainer backed by a TF SavedModel."""

    def __init__(self, model_path: str):
        import tensorflow as tf

        self._tf   = tf
        self._lock = threading.Lock()

        # Load SavedModel — works with Keras 3 / TF 2.16+
        self._sm    = tf.saved_model.load(model_path)
        self._infer = self._sm.signatures["serving_default"]

        # Discover the input / output tensor keys from the signature.
        self._in_key  = list(self._infer.structured_input_signature[1].keys())[0]
        dummy_out     = self._infer(**{self._in_key: tf.zeros([1, 224, 224, 3])})
        self._out_key = list(dummy_out.keys())[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        input_array: np.ndarray,
        class_idx: int,
        original_image: Image.Image,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Compute a gradient saliency map and return it overlaid on original_image.

        Args:
            input_array    : np.ndarray  (1, 224, 224, 3) float32  preprocessed
            class_idx      : int  predicted class index
            original_image : PIL.Image  original (un-preprocessed) image
            alpha          : heatmap blend weight (0 = invisible, 1 = only heatmap)

        Returns:
            PIL.Image with saliency overlay at original resolution
        """
        with self._lock:
            heatmap = self._compute_saliency(input_array, class_idx)
        return self._overlay(heatmap, original_image, alpha)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_saliency(self, input_array: np.ndarray, class_idx: int) -> np.ndarray:
        """Return a float32 saliency map in [0, 1] at 224×224 resolution."""
        tf = self._tf

        # tf.Variable is automatically watched by GradientTape.
        input_var = tf.Variable(input_array.astype(np.float32))

        with tf.GradientTape() as tape:
            outputs = self._infer(**{self._in_key: input_var})
            preds   = outputs[self._out_key]
            loss    = preds[:, class_idx]

        grads = tape.gradient(loss, input_var)   # (1, 224, 224, 3)

        if grads is None:
            raise RuntimeError(
                "Gradient computation returned None — the model's serving_default "
                "signature may not be differentiable with GradientTape.  "
                "Try a different call_endpoint."
            )

        # Gradient × Input: highlights where the gradient AND the activation are both large.
        grad_input = tf.abs(grads[0] * input_var[0]).numpy()   # (224, 224, 3)
        saliency   = np.mean(grad_input, axis=-1)               # (224, 224)

        # Normalise to [0, 1]
        saliency = np.maximum(saliency, 0)
        max_val  = saliency.max()
        if max_val > 1e-8:
            saliency /= max_val

        return saliency.astype(np.float32)

    def _overlay(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        alpha: float,
    ) -> Image.Image:
        """Resize heatmap to original_image size and blend with JET colormap."""
        orig_w, orig_h  = original_image.size
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        orig_array = np.array(original_image, dtype=np.uint8)
        overlay    = cv2.addWeighted(orig_array, 1.0 - alpha, heatmap_rgb, alpha, 0)

        return Image.fromarray(overlay)
