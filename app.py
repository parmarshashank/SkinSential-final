from __future__ import annotations

"""
app.py — Main Tkinter UI for the Skin Disease Classifier.

Layout (responsive — stretches to any screen size)
------
  Title + subtitle
  ┌─────────────────────────────────────┐  ← expands with window
  │           Image Preview             │
  └─────────────────────────────────────┘
  [Capture Webcam]      [Upload Image]
           [ Predict ]
  ┌─────────────────────────────────────┐
  │  Class:       —   Confidence:  —   │
  └─────────────────────────────────────┘
     [ Explain (Grad-CAM)  ]
  ─────── status bar ───────────────────
"""

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import cv2
import numpy as np
from PIL import Image, ImageTk

from inference import TFLiteClassifier
from gradcam import GradCAMExplainer
from utils import preprocess_image, resize_for_display
import config

_DIR          = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL  = os.path.join(_DIR, config.TFLITE_MODEL_FILE)
TF_SAVEDMODEL = os.path.join(_DIR, config.TF_SAVEDMODEL_FILE)

BG          = "#F0F2F5"
CARD_BG     = "#FFFFFF"
IMG_BG      = "#D8DCE4"
TEXT_DARK   = "#1B1B2F"
TEXT_MID    = "#4A4A6A"
TEXT_LIGHT  = "#9A9AB0"
ACCENT_LINE = "#C8CDD8"

BTN_BLUE   = "#1A73E8"
BTN_GREEN  = "#1E8C45"
BTN_ORANGE = "#C0392B"
BTN_GREY   = "#4A5568"
BTN_FG     = "#FFFFFF"

FONT_TITLE  = ("Helvetica", 18, "bold")
FONT_SUB    = ("Helvetica", 9)
FONT_LABEL  = ("Helvetica", 11)
FONT_RES    = ("Helvetica", 13, "bold")
FONT_CONF   = ("Helvetica", 12)
FONT_BTN    = ("Helvetica", 11, "bold")
FONT_STATUS = ("Helvetica", 9)

MIN_W, MIN_H = 420, 560
DEF_W, DEF_H = 520, 700

class SkinClassifierApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._configure_window()

        self.current_image: Image.Image | None   = None
        self.input_array:   np.ndarray | None    = None
        self.last_class_idx: int | None          = None
        self._photo_ref: ImageTk.PhotoImage | None = None

        self._load_tflite()
        self._gradcam: GradCAMExplainer | None = None

        self._camera_source = self._detect_default_source()

        self._build_ui()
        self._refresh_source_label()
        self._set_status("Ready — load an image to begin.")

    def _configure_window(self):
        self.root.title("Skin Disease Classifier")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.minsize(config.MIN_WIDTH, config.MIN_HEIGHT)
        self.root.resizable(True, True)
        self.root.configure(bg=BG)

    def _load_tflite(self):
        try:
            self.classifier = TFLiteClassifier(TFLITE_MODEL)
        except Exception as exc:
            messagebox.showerror("Model Error",
                f"Failed to load TFLite model:\n{TFLITE_MODEL}\n\n{exc}")
            raise SystemExit(1) from exc

    def _detect_default_source(self):

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return 0
        return config.DEFAULT_STREAM_URL

    def _source_label(self) -> str:
        s = self._camera_source
        if isinstance(s, int):
            return f"Camera {s}  (local)"
        return s

    def _refresh_source_label(self):
        self._src_var.set(f"  Source: {self._source_label()}")

    @property
    def gradcam(self) -> GradCAMExplainer:
        if self._gradcam is None:
            self._set_status("Loading TF model for Grad-CAM (one-time)…")
            self.root.update_idletasks()
            self._gradcam = GradCAMExplainer(TF_SAVEDMODEL)
        return self._gradcam

    def _build_ui(self):
        root = self.root

        # Landscape two-column split: controls left, image right
        root.columnconfigure(0, weight=0)  # left panel — fixed width
        root.columnconfigure(1, weight=1)  # right panel — expands
        root.rowconfigure(0, weight=1)

        # ── Left panel ────────────────────────────────────────────────
        left = tk.Frame(root, bg=BG, width=192)
        left.pack_propagate(False)          # hold fixed width regardless of children
        left.grid(row=0, column=0, sticky="nsew")

        tk.Label(left, text="Skin Classifier",
                 font=("Helvetica", 11, "bold"), bg=BG, fg=TEXT_DARK,
        ).pack(pady=(10, 1), padx=8)

        tk.Label(left, text="Offline · AI-powered",
                 font=("Helvetica", 7), bg=BG, fg=TEXT_LIGHT,
        ).pack(pady=(0, 5))

        tk.Frame(left, bg=ACCENT_LINE, height=1).pack(fill="x")

        # Source bar
        src_bar = tk.Frame(left, bg="#E2E6EF")
        src_bar.pack(fill="x", pady=(3, 3))

        self._src_var = tk.StringVar()
        tk.Label(src_bar, textvariable=self._src_var,
                 font=("Helvetica", 7), bg="#E2E6EF", fg=TEXT_MID,
                 anchor="w", padx=4, pady=2,
        ).pack(side="left", fill="x", expand=True)

        _ColorButton(src_bar, "⚙", self._open_camera_settings,
                     "#5A6270", font=("Helvetica", 9, "bold"),
                     padx=8, pady=3,
        ).pack(side="right", padx=2, pady=2)

        tk.Frame(left, bg=ACCENT_LINE, height=1).pack(fill="x")

        # Buttons
        btn_area = tk.Frame(left, bg=BG)
        btn_area.pack(fill="x", padx=8, pady=(8, 0))

        self._btn_capture = _ColorButton(
            btn_area, "Capture Image", self.capture_image, BTN_BLUE,
            font=("Helvetica", 10, "bold"), padx=4, pady=11)
        self._btn_capture.pack(fill="x", pady=(0, 5))

        self._btn_upload = _ColorButton(
            btn_area, "Upload Image", self.upload_image, BTN_GREY,
            font=("Helvetica", 10, "bold"), padx=4, pady=11)
        self._btn_upload.pack(fill="x", pady=(0, 7))

        tk.Frame(btn_area, bg=ACCENT_LINE, height=1).pack(fill="x", pady=(0, 7))

        self._btn_predict = _ColorButton(
            btn_area, "Predict", self.predict, BTN_GREEN,
            font=("Helvetica", 12, "bold"), padx=4, pady=13)
        self._btn_predict.pack(fill="x", pady=(0, 7))
        self._btn_predict.set_enabled(False)

        # Result card
        result_border = tk.Frame(left, bg=ACCENT_LINE)
        result_border.pack(fill="x", padx=8, pady=(0, 7))

        result_card = tk.Frame(result_border, bg=CARD_BG, padx=8, pady=7)
        result_card.pack(fill="both", padx=1, pady=1)

        self._lbl_class = tk.Label(
            result_card, text="Class:  —",
            font=("Helvetica", 10, "bold"), bg=CARD_BG, fg=TEXT_DARK, anchor="w")
        self._lbl_class.pack(anchor="w")

        self._lbl_confidence = tk.Label(
            result_card, text="Conf:  —",
            font=("Helvetica", 9), bg=CARD_BG, fg=TEXT_MID, anchor="w")
        self._lbl_confidence.pack(anchor="w")

        # Explain button
        self._btn_explain = _ColorButton(
            left, "Explain (Grad-CAM)", self.explain, BTN_ORANGE,
            font=("Helvetica", 9, "bold"), padx=4, pady=11)
        self._btn_explain.pack(fill="x", padx=8, pady=(0, 6))
        self._btn_explain.set_enabled(False)

        # Spacer pushes status bar to bottom
        tk.Frame(left, bg=BG).pack(fill="both", expand=True)

        # Status bar pinned to bottom of left panel
        self._status_var = tk.StringVar()
        tk.Label(left, textvariable=self._status_var,
                 font=("Helvetica", 7), bg="#C8CDD8", fg=TEXT_MID,
                 anchor="w", padx=6, pady=4,
                 wraplength=180, justify="left",
        ).pack(fill="x", side="bottom")

        # ── Right panel: image preview (fills remaining space) ────────
        preview_border = tk.Frame(root, bg=ACCENT_LINE)
        preview_border.grid(row=0, column=1, sticky="nsew",
                            padx=(0, 4), pady=4)
        preview_border.rowconfigure(0, weight=1)
        preview_border.columnconfigure(0, weight=1)

        self._preview_frame = tk.Frame(preview_border, bg=IMG_BG)
        self._preview_frame.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        self._preview_frame.rowconfigure(0, weight=1)
        self._preview_frame.columnconfigure(0, weight=1)

        self._img_label = tk.Label(
            self._preview_frame,
            text="No image\n\nCapture or\nupload",
            font=FONT_LABEL, bg=IMG_BG, fg=TEXT_LIGHT, justify="center",
        )
        self._img_label.grid(row=0, column=0, sticky="nsew")

        self._preview_frame.bind("<Configure>", self._on_preview_resize)

    def capture_image(self):

        source = self._camera_source
        label  = self._source_label()
        self._set_status(f"Capturing from {label}…")
        self._btn_capture.set_enabled(False)
        self.root.update_idletasks()

        def _worker():
            try:
                pil = _capture_frame(source)
                self.root.after(0, lambda: self._load_pil(pil))
                self.root.after(0, lambda: self._set_status(
                    f"Captured — click Predict."))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Camera Error", str(exc)))
                self.root.after(0, lambda: self._set_status("Capture failed."))
            finally:
                self.root.after(0, lambda: self._btn_capture.set_enabled(True))

        threading.Thread(target=_worker, daemon=True).start()

    def _open_camera_settings(self):

        dlg = _CameraSettingsDialog(self.root, self._camera_source)
        self.root.wait_window(dlg)
        if dlg.result is not None:
            self._camera_source = dlg.result
            self._refresh_source_label()

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self._load_pil(Image.open(path).convert("RGB"))
            self._set_status(f"Loaded: {os.path.basename(path)} — click Predict.")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Cannot open image:\n{exc}")
            self._set_status("Image load failed.")

    def predict(self):
        if self.input_array is None:
            return
        self._set_status("Running inference…")
        self.root.update_idletasks()
        try:
            class_name, confidence, class_idx = self.classifier.predict(self.input_array)
            self.last_class_idx = class_idx
            self._lbl_class.config(text=f"Class:  {class_name.capitalize()}")
            self._lbl_confidence.config(text=f"Conf:  {confidence:.1f}%")
            self._btn_explain.set_enabled(True)
            self._set_status(f"Prediction: {class_name.capitalize()}  ({confidence:.1f}%)")
        except Exception as exc:
            messagebox.showerror("Inference Error", str(exc))
            self._set_status("Inference failed.")

    def explain(self):
        if self.input_array is None or self.last_class_idx is None:
            return
        self._btn_explain.set_enabled(False)
        self._btn_predict.set_enabled(False)
        self._set_status("Generating saliency map — please wait…")

        def _worker():
            try:
                overlay = self.gradcam.explain(
                    self.input_array, self.last_class_idx, self.current_image)
                self.root.after(0, lambda: self._show_overlay(overlay))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Explain Error", str(exc)))
                self.root.after(0, lambda: self._set_status("Explain failed."))
            finally:
                self.root.after(0, lambda: self._btn_explain.set_enabled(True))
                self.root.after(0, lambda: self._btn_predict.set_enabled(True))

        threading.Thread(target=_worker, daemon=True).start()

    def _load_pil(self, pil: Image.Image):
        self.current_image  = pil
        self.input_array, _ = preprocess_image(pil)
        self.last_class_idx = None
        self._update_preview(pil)
        self._btn_predict.set_enabled(True)
        self._btn_explain.set_enabled(False)
        self._lbl_class.config(text="Class:  —")
        self._lbl_confidence.config(text="Conf:  —")

    def _update_preview(self, pil: Image.Image):
        w = self._preview_frame.winfo_width()  or 400
        h = self._preview_frame.winfo_height() or 280
        display = resize_for_display(pil, (w - 4, h - 4))
        self._photo_ref = ImageTk.PhotoImage(display)
        self._img_label.config(image=self._photo_ref, text="",
                               bg=IMG_BG, anchor="center")

    def _on_preview_resize(self, _event):
        if self.current_image is not None:
            self._update_preview(self.current_image)

    def _show_overlay(self, overlay: Image.Image):
        self._update_preview(overlay)
        self._set_status("Saliency overlay ready — warm regions drove the prediction.")

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")
        self.root.update_idletasks()

def _scan_cameras(max_idx: int = 6) -> list[int]:

    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found

def _url_error_message(url: str, exc: Exception) -> str:

    import urllib.error
    reason = ""
    if hasattr(exc, "reason"):
        reason = str(exc.reason).lower()
    msg = str(exc).lower()
    combined = reason + msg

    if "refused" in combined:
        return (
            f"Connection refused — the camera app server is not running.\n"
            "→ Open the app and tap  Start Server / Start."
        )
    if "timed out" in combined or "time out" in combined or "timeout" in combined:
        return (
            "Connection timed out.\n"
            "→ Check that the IP address is correct.\n"
            "→ Make sure your phone and this device are on the same WiFi."
        )
    if "name or service not known" in combined or "nodename" in combined:
        return "Hostname not found — paste an IP address (e.g. 192.168.1.42), not a name."
    return f"Network error: {exc}"

def _grab_mjpeg_frame(url: str, timeout: int = 8) -> Image.Image:

    import urllib.request

    import urllib.error
    req = urllib.request.Request(url, headers={"Connection": "close"})
    try:
        stream_ctx = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                "The camera app requires a username and password.\n\n"
                "Add them to the URL like this:\n"
                "  http://username:password@172.20.10.1:8080/live\n\n"
                "→ Check the app Settings for the credentials\n"
                "  (IP Camera Lite: gear icon → Username / Password)."
            ) from exc
        raise RuntimeError(f"HTTP {exc.code} error from stream: {exc.reason}") from exc

    with stream_ctx as stream:
        buf = b""

        for _ in range(512):
            buf += stream.read(4096)
            s = buf.find(b"\xff\xd8")
            e = buf.find(b"\xff\xd9", s + 2) if s != -1 else -1
            if s != -1 and e != -1:
                jpeg = buf[s: e + 2]
                arr  = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    raise RuntimeError(
        "Connected to the stream but could not find a JPEG frame.\n"
        "→ Make sure the app is actively streaming video."
    )

def _fetch_snapshot(url: str, timeout: int = 6) -> Image.Image:

    import urllib.request
    import urllib.error
    import socket

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                "The camera app requires a username and password.\n\n"
                "Add them to the URL like this:\n"
                "  http://username:password@172.20.10.1:8080/live\n\n"
                "→ Check the app Settings for the credentials\n"
                "  (IP Camera Lite: gear icon → Username / Password)."
            ) from exc
        raise RuntimeError(f"HTTP {exc.code} error from camera: {exc.reason}") from exc
    except (urllib.error.URLError, socket.timeout) as exc:
        raise RuntimeError(_url_error_message(url, exc)) from exc

    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:

        try:
            return _grab_mjpeg_frame(url, timeout=timeout)
        except Exception:
            pass
        raise RuntimeError(
            "The URL responded but did not return a usable image.\n"
            "→ Try the stream URL instead (e.g. /live, /video, /stream)."
        )
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def _test_connection(url: str, timeout: int = 6) -> str:

    import urllib.request
    import urllib.error
    import socket

    try:

        req = urllib.request.Request(url, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(256)
        return ""
    except (urllib.error.URLError, socket.timeout) as exc:
        return _url_error_message(url, exc)
    except Exception as exc:
        return str(exc)

def _capture_frame(source) -> Image.Image:

    import urllib.error, socket

    if isinstance(source, str):
        url_lower = source.lower().split("?")[0]

        if url_lower.startswith("rtsp://"):

            for backend in (cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY):
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    break
                cap.release()
            else:

                http_url = "http://" + source[len("rtsp://"):]
                try:
                    return _grab_mjpeg_frame(http_url)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Could not open RTSP stream:\n{source}\n\n"
                    "OpenCV on this machine may not have RTSP support.\n\n"
                    "→ Use WiFi hotspot (not USB) — connect Mac to iPhone hotspot,\n"
                    "  then use  http://172.20.10.1:<port>/live  instead of RTSP.\n"
                    "→ Or check if the app has an HTTP stream URL."
                )
            try:
                for _ in range(config.IP_STREAM_WARMUP_FRAMES):
                    cap.read()
                ret, frame = cap.read()
            finally:
                cap.release()
            if not ret or frame is None:
                raise RuntimeError("RTSP stream opened but no frame received.")
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if url_lower.startswith("http://") or url_lower.startswith("https://"):
            if url_lower.endswith((".jpg", ".jpeg")):
                return _fetch_snapshot(source)

            try:
                return _grab_mjpeg_frame(source)
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(_url_error_message(source, exc)) from exc

        raise RuntimeError(
            f"Unrecognised URL format:\n{source}\n\n"
            "Supported: http://, https://, rtsp://"
        )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open local camera device {source}.")
    try:
        cap.read()
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Camera {source} opened but no frame received.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

class _NumpadDialog(tk.Toplevel):

    _W, _H = 320, 380

    def __init__(self, parent, prefill: str = ""):
        super().__init__(parent)
        self.title("Enter IP Address")
        self.geometry(f"{self._W}x{self._H}")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.result = None

        self._val = tk.StringVar(value=prefill)
        self._build()

        px = parent.winfo_x() + parent.winfo_width()  // 2
        py = parent.winfo_y() + parent.winfo_height() // 2
        self.geometry(f"{self._W}x{self._H}+{px - self._W//2}+{py - self._H//2}")

        self.update_idletasks()
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass

    def _build(self):

        disp_frame = tk.Frame(self, bg=ACCENT_LINE, bd=1)
        disp_frame.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(disp_frame, textvariable=self._val,
                 font=("Courier", 16, "bold"),
                 bg=CARD_BG, fg=TEXT_DARK, anchor="e", padx=8, pady=8,
        ).pack(fill="x", padx=1, pady=1)

        grid = tk.Frame(self, bg=BG)
        grid.pack(padx=16, pady=(0, 10), fill="both", expand=True)

        ROWS = [
            ["7", "8", "9"],
            ["4", "5", "6"],
            ["1", "2", "3"],
            [".", "0", "⌫"],
        ]
        for r, row in enumerate(ROWS):
            for c, key in enumerate(row):
                cmd = (lambda k=key: self._backspace()) if key == "⌫" \
                      else (lambda k=key: self._press(k))
                bg = "#D0D8E8" if key == "⌫" else CARD_BG
                btn = tk.Label(grid, text=key, font=("Helvetica", 18, "bold"),
                               bg=bg, fg=TEXT_DARK, relief="flat",
                               cursor="hand2", pady=0)
                btn.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")
                btn.bind("<Button-1>", lambda _, fn=cmd: fn())
            grid.rowconfigure(r, weight=1)
        for c in range(3):
            grid.columnconfigure(c, weight=1)

        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", padx=16, pady=(0, 14))
        _ColorButton(bot, "Clear", self._clear, BTN_GREY,
                     padx=16, pady=8).pack(side="left", expand=True, fill="x", padx=(0, 4))
        _ColorButton(bot, "  OK  ", self._confirm, BTN_GREEN,
                     padx=16, pady=8).pack(side="left", expand=True, fill="x", padx=(4, 0))

    def _press(self, key: str):
        self._val.set(self._val.get() + key)

    def _backspace(self):
        self._val.set(self._val.get()[:-1])

    def _clear(self):
        self._val.set("")

    def _confirm(self):
        self.result = self._val.get()
        self.destroy()

class _CameraSettingsDialog(tk.Toplevel):

    _W, _H = 480, 360

    def __init__(self, parent: tk.Tk, current_source):
        super().__init__(parent)
        self.title("Camera Settings")
        self.geometry(f"{self._W}x{self._H}")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.result = None

        self._current  = current_source
        self._mode     = tk.StringVar(value="ip" if isinstance(current_source, str) else "local")
        self._idx_var  = tk.IntVar(value=current_source if isinstance(current_source, int) else 0)
        self._url_var  = tk.StringVar(value=current_source if isinstance(current_source, str)
                                      else config.DEFAULT_STREAM_URL)
        self._cameras: list[int] = []

        self._build()
        px = parent.winfo_x() + parent.winfo_width()  // 2
        py = parent.winfo_y() + parent.winfo_height() // 2
        self.geometry(f"{self._W}x{self._H}+{px - self._W//2}+{py - self._H//2}")

        self.update_idletasks()
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass

        threading.Thread(target=self._do_scan, daemon=True).start()

    def _build(self):
        tk.Label(self, text="Camera Settings",
                 font=("Helvetica", 13, "bold"), bg=BG, fg=TEXT_DARK,
        ).pack(pady=(14, 10))

        tk.Frame(self, bg=ACCENT_LINE, height=1).pack(fill="x", padx=20)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        local_row = tk.Frame(body, bg=BG)
        local_row.pack(fill="x", pady=(0, 6))

        tk.Radiobutton(local_row, text="Local camera:",
                       variable=self._mode, value="local",
                       bg=BG, fg=TEXT_DARK, font=("Helvetica", 11),
                       activebackground=BG, selectcolor=BG,
                       command=self._on_mode_change,
        ).pack(side="left")

        self._cam_menu_var = tk.StringVar()
        self._cam_menu = tk.OptionMenu(local_row, self._cam_menu_var, "Scanning…")
        self._cam_menu.config(font=("Helvetica", 11), bg=CARD_BG,
                              relief="flat", highlightthickness=1,
                              highlightbackground=ACCENT_LINE)
        self._cam_menu.pack(side="left", padx=(8, 0))

        tk.Frame(body, bg=ACCENT_LINE, height=1).pack(fill="x", pady=8)

        ip_row = tk.Frame(body, bg=BG)
        ip_row.pack(fill="x", pady=(0, 4))

        tk.Radiobutton(ip_row, text="IP / WiFi stream:",
                       variable=self._mode, value="ip",
                       bg=BG, fg=TEXT_DARK, font=("Helvetica", 11),
                       activebackground=BG, selectcolor=BG,
                       command=self._on_mode_change,
        ).pack(side="left")

        url_frame = tk.Frame(body, bg=BG)
        url_frame.pack(fill="x")

        ef = tk.Frame(url_frame, bg=ACCENT_LINE, bd=1)
        ef.pack(side="left", fill="x", expand=True)
        self._url_entry = tk.Entry(ef, textvariable=self._url_var,
                                   font=("Helvetica", 11), relief="flat", bd=5,
                                   bg=CARD_BG, fg=TEXT_DARK,
                                   insertbackground=TEXT_DARK)
        self._url_entry.pack(fill="x")

        _ColorButton(url_frame, "IP", self._open_numpad, BTN_BLUE,
                     font=("Helvetica", 10, "bold"), padx=10, pady=5,
        ).pack(side="left", padx=(6, 0))

        test_row = tk.Frame(body, bg=BG)
        test_row.pack(fill="x", pady=(8, 0))

        _ColorButton(test_row, "Test Connection", self._test_conn,
                     BTN_BLUE, font=("Helvetica", 10, "bold"), padx=12, pady=5,
        ).pack(side="left")

        self._test_lbl = tk.Label(test_row, text="",
                                  font=("Helvetica", 10), bg=BG, fg=TEXT_MID,
                                  wraplength=280, justify="left")
        self._test_lbl.pack(side="left", padx=(10, 0))

        tk.Frame(self, bg=ACCENT_LINE, height=1).pack(fill="x", padx=20)
        bar = tk.Frame(self, bg=BG, pady=10)
        bar.pack(fill="x")
        _ColorButton(bar, "Cancel", self._cancel, BTN_GREY,
                     padx=16, pady=8).pack(side="right", padx=(4, 20))
        _ColorButton(bar, "  Save  ", self._confirm, BTN_GREEN,
                     font=("Helvetica", 11, "bold"), padx=16, pady=8,
        ).pack(side="right", padx=4)

        self._on_mode_change()

    def _on_mode_change(self):
        if self._mode.get() == "ip":
            self._url_entry.config(state="normal")
            self._cam_menu.config(state="disabled")
        else:
            self._url_entry.config(state="disabled")
            self._cam_menu.config(state="normal")

    def _do_scan(self):
        cameras = _scan_cameras()
        self._cameras = cameras
        self.after(0, self._update_menu)

    def _update_menu(self):
        menu = self._cam_menu["menu"]
        menu.delete(0, "end")

        if not self._cameras:
            self._cam_menu_var.set("No cameras found")
            return

        NAMES = {0: "Camera 0 (built-in)"}
        for i in self._cameras:
            label = NAMES.get(i, f"Camera {i} (external)")
            menu.add_command(label=label,
                             command=lambda v=i, l=label: (self._idx_var.set(v),
                                                           self._cam_menu_var.set(l)))

        first = self._cameras[0]

        if isinstance(self._current, int) and self._current in self._cameras:
            first = self._current
        label = NAMES.get(first, f"Camera {first} (external)")
        self._idx_var.set(first)
        self._cam_menu_var.set(label)

    def _open_numpad(self):
        import re
        url = self._url_var.get()
        m   = re.search(r"://([0-9.]+)", url)
        dlg = _NumpadDialog(self, m.group(1) if m else "")
        self.wait_window(dlg)
        if dlg.result:
            new_ip = dlg.result.strip()
            self._url_var.set(
                url[:m.start(1)] + new_ip + url[m.end(1):]
                if m else f"http://{new_ip}:8081/video"
            )

    def _test_conn(self):
        url = self._url_var.get().strip()
        self._test_lbl.config(text="Testing…", fg=TEXT_MID)
        self.update_idletasks()

        def _worker():
            err = _test_connection(url)
            self.after(0, lambda: self._test_lbl.config(
                text="✓ Connected!" if not err else f"✗ {err.splitlines()[0]}",
                fg="#1E8C45" if not err else "#C0392B",
            ))

        threading.Thread(target=_worker, daemon=True).start()

    def _confirm(self):
        if self._mode.get() == "ip":
            self.result = self._url_var.get().strip() or config.DEFAULT_STREAM_URL
        else:
            self.result = self._idx_var.get()
        self.destroy()

    def _cancel(self):
        self.destroy()

class _ColorButton(tk.Frame):

    def __init__(self, parent, text: str, command, bg: str,
                 fg: str = BTN_FG, font=FONT_BTN,
                 padx: int = 18, pady: int = 8):
        super().__init__(parent, bg=bg, cursor="hand2")
        self._cmd      = command
        self._bg       = bg
        self._fg       = fg
        self._hover_bg = _darken(bg, 0.82)
        self._enabled  = True

        self._lbl = tk.Label(
            self, text=text, bg=bg, fg=fg,
            font=font, padx=padx, pady=pady,
        )
        self._lbl.pack()

        for w in (self, self._lbl):
            w.bind("<Button-1>", self._on_click)
            w.bind("<Enter>",    self._on_enter)
            w.bind("<Leave>",    self._on_leave)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self._lbl.config(bg=self._bg, fg=self._fg, cursor="hand2")
            self.config(bg=self._bg, cursor="hand2")
        else:
            disabled_bg = "#A0A8B4"
            self._lbl.config(bg=disabled_bg, fg="#E0E0E0", cursor="")
            self.config(bg=disabled_bg, cursor="")

    def _on_click(self, _event=None):
        if self._enabled:
            self._cmd()

    def _on_enter(self, _event=None):
        if self._enabled:
            self._lbl.config(bg=self._hover_bg)
            self.config(bg=self._hover_bg)

    def _on_leave(self, _event=None):
        if self._enabled:
            self._lbl.config(bg=self._bg)
            self.config(bg=self._bg)

def _darken(hex_color: str, factor: float) -> str:
    r = max(0, int(int(hex_color[1:3], 16) * factor))
    g = max(0, int(int(hex_color[3:5], 16) * factor))
    b = max(0, int(int(hex_color[5:7], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"

def main():
    root = tk.Tk()
    SkinClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
