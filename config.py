"""
config.py — All user-configurable settings in one place.

Edit this file to change behaviour without touching any other code.
"""

# ── Preprocessing ────────────────────────────────────────────────────
# Controls how pixel values are scaled before being fed to the model.
# Change this if predictions are wrong (see utils.py for details).
#
#   "raw"      →  float32  [0, 255]   model has built-in rescaling  ← default
#   "simple"   →  float32  [0, 1]     model expects pre-scaled input
#   "tf"       →  float32  [-1, 1]    legacy EfficientNet tf-mode
#   "imagenet" →  ImageNet mean/std   PyTorch-style transfer learning
NORM_MODE = "raw"

# ── Camera ───────────────────────────────────────────────────────────
# Default local camera device index used when the dialog opens.
DEFAULT_CAMERA_INDEX = 0

# Default IP stream URL shown in the iPhone-over-WiFi tab.
# Replace with the actual URL shown in your iPhone app.
# Common formats:
#   IP Camera Lite : http://192.168.x.x:8080/shot.jpg   ← snapshot, fastest
#   IP Camera Lite : http://192.168.x.x:8080/video      ← MJPEG stream alternative
#   EpocCam        : http://192.168.x.x:1900/live
#   DroidCam       : http://192.168.x.x:4747/video
#   RTSP (generic) : rtsp://192.168.x.x:8554/stream
DEFAULT_STREAM_URL = "http://100.100.0.122:8081/video"

# Number of warm-up frames to read from an IP stream before grabbing.
# Increase if the first captured frame is dark / blurry.
IP_STREAM_WARMUP_FRAMES = 5

# ── Model paths ──────────────────────────────────────────────────────
# Relative to the directory containing app.py.
TFLITE_MODEL_FILE  = "model_fp16.tflite"
TF_SAVEDMODEL_DIR  = "skinsential_model"

# ── Classes ──────────────────────────────────────────────────────────
# Must match the output order of the model.
# Must match the alphabetical order that image_dataset_from_directory assigns.
# (0=melanoma, 1=normal, 2=psoriasis, 3=ringworm)
CLASSES = ["melanoma", "normal", "psoriasis", "ringworm"]

# ── UI ───────────────────────────────────────────────────────────────
# Default window size (pixels).  The window is resizable; this is just
# the starting size.
WINDOW_WIDTH  = 520
WINDOW_HEIGHT = 700

# Minimum window size — important for Pi's smaller screen.
MIN_WIDTH  = 420
MIN_HEIGHT = 560
