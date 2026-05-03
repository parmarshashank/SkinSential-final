# Skin Disease Classifier

Offline desktop app for classifying skin conditions using a pre-trained EfficientNetB0 model.  
Runs entirely on-device — no internet required.

**Classes:** Melanoma · Psoriasis · Ringworm · Normal

---

## Project Structure

```
proj/
├── app.py                  # Main UI (Tkinter)
├── inference.py            # TFLite fast inference
├── gradcam.py              # Grad-CAM explanation (TF SavedModel)
├── utils.py                # Image preprocessing
├── model_fp16.tflite       # Fast inference model  ← you provide
├── skinsential_model/      # TF SavedModel for Grad-CAM  ← you provide
├── requirements_mac.txt    # Mac / dev dependencies
└── requirements_pi.txt     # Raspberry Pi 4 dependencies
```

---

## Running on Mac (development)

> **macOS 26 Tahoe note:** PyPI `opencv-python` wheels 4.9+ crash on macOS 26
> due to an internal version-check bug. Install OpenCV via Homebrew instead —
> the steps below handle this automatically.

### 1. Install system dependencies via Homebrew

```bash
brew install python@3.10 python-tk@3.10 opencv
```

`opencv` from Homebrew compiles against the current OS and avoids the PyPI
wheel crash entirely.

### 2. Create virtual environment

```bash
cd /path/to/proj
# Remove any old broken venv first
rm -rf .venv

# Use Homebrew's Python 3.10
/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Link Homebrew opencv into the venv

```bash
# Find where Homebrew installed the cv2 module
BREW_CV2=$(ls /opt/homebrew/lib/python3.*/site-packages/cv2* 2>/dev/null | head -1)

# Symlink it into the active venv
ln -s "$BREW_CV2" .venv/lib/python3.10/site-packages/
```

### 4. Install remaining Python packages

```bash
pip install -r requirements_mac.txt
```

### 5. Launch the app

```bash
python app.py
```

---

## Running on Raspberry Pi 4

### 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-tk libatlas-base-dev libhdf5-dev \
                    libopenblas-dev python3.10 python3.10-venv
```

### 2. Install tflite-runtime wheel

The standard PyPI `tflite-runtime` wheel works on Pi OS Bookworm (aarch64).

```bash
pip install tflite-runtime
```

If it fails, download a prebuilt wheel from:
https://github.com/google-coral/pycoral/releases  
(pick the one matching your Python version and OS)

```bash
pip install tflite_runtime-*.whl
```

### 3. Set up environment

```bash
cd ~/proj
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_pi.txt
```

### 4. Launch

```bash
python app.py
```

> **Note:** Grad-CAM ("Explain") is disabled on Pi by default because it requires
> the full TensorFlow package, which is heavy. To enable it on Pi, install
> `tensorflow` instead of `tflite-runtime`. For production Pi deployments,
> leave Grad-CAM unused and rely on TFLite for fast predictions.

---

## Preprocessing Details

The app uses **EfficientNetB0 torch-mode preprocessing** (matches Keras default):

```
1. Resize to 224 × 224
2. Divide by 255  →  [0, 1]
3. Subtract ImageNet mean  [0.485, 0.456, 0.406]
4. Divide by ImageNet std  [0.229, 0.224, 0.225]
```

If your model was trained with different normalization (e.g. `/127.5 - 1`),
edit `_MEAN` / `_STD` in `utils.py` accordingly.

---

## UX Flow

| Step | Action | What happens |
|------|--------|-------------|
| 1 | Click **Capture Webcam** | Takes a single frame from camera |
| 2 | — or — **Upload Image** | Opens file picker |
| 3 | Click **Predict** | TFLite runs (<200 ms on Pi), shows class + confidence |
| 4 | Click **Explain** | TF SavedModel computes Grad-CAM; heatmap overlaid on image |

---

## Performance Notes

- TFLite model is loaded **once** at startup and reused.
- TF SavedModel is loaded **lazily** on the first Explain click.
- Grad-CAM runs in a **background thread** — UI stays responsive.
- Expected latency on Pi 4: `<200 ms` for TFLite inference.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `zsh: command not found: python3.10` | `brew install python@3.10` |
| `No module named 'tkinter'` | `brew install python-tk@3.10` (Mac) or `sudo apt install python3-tk` (Pi) |
| `macOS 26 … or later required` crash | Do **not** `pip install opencv-python`; use `brew install opencv` + symlink (see setup steps above) |
| `No module named 'cv2'` | Symlink step missing — re-run the `ln -s` command from step 3 |
| Camera not detected | Check System Settings → Privacy → Camera; try `cv2.VideoCapture(1)` if multiple cameras |
| `ValueError` on Grad-CAM layer | Model layer names differ; the fallback in `gradcam.py` auto-detects the last Conv2D |
| Low confidence on all classes | Verify preprocessing matches training normalization (`utils.py`) |
| Slow Grad-CAM | Normal — TF forward + backward pass on CPU takes 3–10 s |
