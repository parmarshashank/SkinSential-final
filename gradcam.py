from __future__ import annotations

"""
gradcam.py — True Grad-CAM using the top_conv layer of EfficientNetB0.

EfficientNetB0 is a nested submodel, so we can't build a single
keras.Model that spans from the outer input to an intermediate layer
inside the submodel (graph boundary). Fix: split into two chained models:

  conv_model : outer_input → top_conv activations
  tail_model : top_conv activations → predictions
                (top_bn → top_activation → gap → bn → fc1 → drop → predictions)

GradientTape watches conv_outputs between the two models, giving true
Grad-CAM gradients.
"""

import threading
import numpy as np
import cv2
from PIL import Image


class GradCAMExplainer:
    """True Grad-CAM explainer backed by a Keras .keras model."""

    def __init__(self, model_path: str):
        import tensorflow as tf
        from tensorflow import keras

        self._tf   = tf
        self._lock = threading.Lock()

        full_model = keras.models.load_model(model_path)
        self._build_grad_models(full_model)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_grad_models(self, full_model):
        from tensorflow import keras

        try:
            # Nested model (correct build): EfficientNetB0 is a submodel
            base      = full_model.get_layer("efficientnetb0")
            last_conv = base.get_layer("top_conv")

            # conv_model: base's own input → top_conv output
            # (base.inputs accepts the same raw [0,255] as the outer model)
            self._conv_model = keras.Model(
                inputs=base.inputs,
                outputs=last_conv.output,
            )

            # tail_model: top_conv output → predictions
            # Re-uses the trained weight tensors from base + outer head layers
            conv_out_shape = self._conv_model.output_shape[1:]   # (H, W, C)
            x = tail_in = keras.Input(shape=conv_out_shape)
            x = base.get_layer("top_bn")(x)
            x = base.get_layer("top_activation")(x)
            x = full_model.get_layer("gap")(x)
            x = full_model.get_layer("bn")(x)
            x = full_model.get_layer("fc1")(x)
            x = full_model.get_layer("drop")(x)
            x = full_model.get_layer("predictions")(x)
            self._tail_model = keras.Model(tail_in, x)

            self._nested = True

        except ValueError:
            # Flat model (old build via input_tensor=): top_conv is directly accessible
            last_conv = full_model.get_layer("top_conv")
            self._flat_grad_model = keras.Model(
                inputs=full_model.inputs,
                outputs=[last_conv.output, full_model.output],
            )
            self._nested = False

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
        Compute Grad-CAM and return it overlaid on original_image.

        Args:
            input_array    : np.ndarray  (1, 224, 224, 3) float32  raw [0, 255]
            class_idx      : int  predicted class index
            original_image : PIL.Image  original image at any resolution
            alpha          : heatmap blend weight

        Returns:
            PIL.Image with Grad-CAM overlay at original_image's resolution
        """
        with self._lock:
            heatmap = self._compute_gradcam(input_array, class_idx)
        return self._overlay(heatmap, original_image, alpha)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_gradcam(self, input_array: np.ndarray, class_idx: int) -> np.ndarray:
        tf  = self._tf
        img = tf.constant(input_array.astype(np.float32))

        if self._nested:
            with tf.GradientTape() as tape:
                # Step 1: input → top_conv activations
                conv_outputs = self._conv_model(img, training=False)
                tape.watch(conv_outputs)
                # Step 2: top_conv activations → predictions
                predictions  = self._tail_model(conv_outputs, training=False)
                class_score  = predictions[:, class_idx]
        else:
            with tf.GradientTape() as tape:
                conv_outputs, predictions = self._flat_grad_model(
                    [img], training=False)
                tape.watch(conv_outputs)
                class_score = predictions[:, class_idx]

        grads   = tape.gradient(class_score, conv_outputs)  # (1, H, W, C)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))     # (C,)
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled, axis=-1)  # (H, W)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
        return heatmap.astype(np.float32)

    def _overlay(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        alpha: float,
    ) -> Image.Image:
        orig_w, orig_h = original_image.size
        hm = cv2.resize(heatmap, (orig_w, orig_h))
        hm = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        orig_array = np.array(original_image.convert("RGB"), dtype=np.uint8)
        overlay    = cv2.addWeighted(orig_array, 1.0 - alpha, hm, alpha, 0)
        return Image.fromarray(overlay)
