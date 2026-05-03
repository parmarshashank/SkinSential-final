# Postponed annotation evaluation — makes X | None / tuple[...] work on Python 3.9.
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

# ------------------------------------------------------------------
# Paths  (resolved from config)
# ------------------------------------------------------------------
_DIR          = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL  = os.path.join(_DIR, config.TFLITE_MODEL_FILE)
TF_SAVEDMODEL = os.path.join(_DIR, config.TF_SAVEDMODEL_DIR)

# ------------------------------------------------------------------
# Palette  (works on Aqua/macOS + Raspbian — Label-based buttons)
# ------------------------------------------------------------------
BG          = "#F0F2F5"
CARD_BG     = "#FFFFFF"
IMG_BG      = "#D8DCE4"
TEXT_DARK   = "#1B1B2F"
TEXT_MID    = "#4A4A6A"
TEXT_LIGHT  = "#9A9AB0"
ACCENT_LINE = "#C8CDD8"

BTN_BLUE   = "#1A73E8"   # Capture
BTN_GREEN  = "#1E8C45"   # Predict
BTN_ORANGE = "#C0392B"   # Explain
BTN_GREY   = "#4A5568"   # Upload
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


# ==================================================================
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

        self._build_ui()
        self._set_status("Ready — load an image to begin.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

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

    @property
    def gradcam(self) -> GradCAMExplainer:
        if self._gradcam is None:
            self._set_status("Loading TF model for Grad-CAM (one-time)…")
            self.root.update_idletasks()
            self._gradcam = GradCAMExplainer(TF_SAVEDMODEL)
        return self._gradcam

    # ------------------------------------------------------------------
    # UI — responsive grid
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self.root

        # Grid: column 0 fills width; row 2 (preview) fills leftover height
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)   # image preview row

        # ── Row 0: title ───────────────────────────────────────────────
        tk.Label(root, text="Skin Disease Classifier",
                 font=FONT_TITLE, bg=BG, fg=TEXT_DARK
        ).grid(row=0, column=0, pady=(16, 2))

        tk.Label(root, text="Offline · AI-powered · Private",
                 font=FONT_SUB, bg=BG, fg=TEXT_LIGHT
        ).grid(row=1, column=0, pady=(0, 8))

        # ── Row 2: image preview (expands) ─────────────────────────────
        preview_border = tk.Frame(root, bg=ACCENT_LINE)
        preview_border.grid(row=2, column=0, padx=24, pady=(0, 8),
                            sticky="nsew")
        preview_border.rowconfigure(0, weight=1)
        preview_border.columnconfigure(0, weight=1)

        self._preview_frame = tk.Frame(preview_border, bg=IMG_BG)
        self._preview_frame.grid(row=0, column=0, padx=1, pady=1, sticky="nsew")
        self._preview_frame.rowconfigure(0, weight=1)
        self._preview_frame.columnconfigure(0, weight=1)

        self._img_label = tk.Label(
            self._preview_frame,
            text="No image loaded\n\nCapture from webcam\nor upload a file",
            font=FONT_LABEL, bg=IMG_BG, fg=TEXT_LIGHT, justify="center",
        )
        self._img_label.grid(row=0, column=0, sticky="nsew")

        # Redraw image whenever the preview frame is resized
        self._preview_frame.bind("<Configure>", self._on_preview_resize)

        # ── Row 3: Capture / Upload ────────────────────────────────────
        btn_row = tk.Frame(root, bg=BG)
        btn_row.grid(row=3, column=0, pady=(0, 6))

        self._btn_capture = _ColorButton(
            btn_row, "Capture Webcam", self.capture_image, BTN_BLUE)
        self._btn_capture.pack(side="left", padx=8)

        self._btn_upload = _ColorButton(
            btn_row, "Upload Image", self.upload_image, BTN_GREY)
        self._btn_upload.pack(side="left", padx=8)

        # ── Row 4: Predict ─────────────────────────────────────────────
        self._btn_predict = _ColorButton(
            root, "        Predict        ", self.predict, BTN_GREEN,
            font=("Helvetica", 13, "bold"), padx=36, pady=10)
        self._btn_predict.grid(row=4, column=0, pady=(0, 8))
        self._btn_predict.set_enabled(False)

        # ── Row 5: Result card ─────────────────────────────────────────
        result_border = tk.Frame(root, bg=ACCENT_LINE)
        result_border.grid(row=5, column=0, padx=24, pady=(0, 8), sticky="ew")

        result_card = tk.Frame(result_border, bg=CARD_BG, padx=18, pady=12)
        result_card.pack(fill="both", padx=1, pady=1)
        result_card.columnconfigure(0, weight=1)
        result_card.columnconfigure(1, weight=1)

        self._lbl_class = tk.Label(
            result_card, text="Class:  —",
            font=FONT_RES, bg=CARD_BG, fg=TEXT_DARK, anchor="w")
        self._lbl_class.grid(row=0, column=0, sticky="w")

        self._lbl_confidence = tk.Label(
            result_card, text="Confidence:  —",
            font=FONT_CONF, bg=CARD_BG, fg=TEXT_MID, anchor="e")
        self._lbl_confidence.grid(row=0, column=1, sticky="e")

        # ── Row 6: Explain ─────────────────────────────────────────────
        self._btn_explain = _ColorButton(
            root, "Explain (Grad-CAM)", self.explain, BTN_ORANGE)
        self._btn_explain.grid(row=6, column=0, pady=(0, 4))
        self._btn_explain.set_enabled(False)

        # ── Row 7: Status bar ──────────────────────────────────────────
        self._status_var = tk.StringVar()
        tk.Label(root, textvariable=self._status_var,
                 font=FONT_STATUS, bg="#C8CDD8", fg=TEXT_MID,
                 anchor="w", padx=10, pady=4
        ).grid(row=7, column=0, sticky="ew")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def capture_image(self):
        """Open the camera-source dialog, then grab one frame."""
        dialog = _CameraDialog(self.root)
        self.root.wait_window(dialog)

        source = dialog.result
        if source is None:
            return  # user cancelled

        label = f"stream {source}" if isinstance(source, str) else f"camera {source}"
        self._set_status(f"Capturing from {label}…")
        self.root.update_idletasks()

        def _worker():
            try:
                pil = _capture_frame(source)
                self.root.after(0, lambda: self._load_pil(pil))
                self.root.after(0, lambda: self._set_status(
                    f"Captured from {label} — click Predict."))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Camera Error", str(exc)))
                self.root.after(0, lambda: self._set_status("Capture failed."))

        threading.Thread(target=_worker, daemon=True).start()

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
            self._lbl_confidence.config(text=f"Confidence:  {confidence:.1f}%")
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pil(self, pil: Image.Image):
        self.current_image  = pil
        self.input_array, _ = preprocess_image(pil)
        self.last_class_idx = None
        self._update_preview(pil)
        self._btn_predict.set_enabled(True)
        self._btn_explain.set_enabled(False)
        self._lbl_class.config(text="Class:  —")
        self._lbl_confidence.config(text="Confidence:  —")

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


# ==================================================================
# Camera helpers
# ==================================================================

def _scan_cameras(max_idx: int = 6) -> list[int]:
    """Return indices of all openable local camera devices."""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
            cap.release()
    return found


def _url_error_message(url: str, exc: Exception) -> str:
    """Turn a network exception into a plain-English hint."""
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
    """
    Read just enough of an HTTP MJPEG stream to extract one JPEG frame.
    More reliable than cv2.VideoCapture for HTTP streams on macOS.
    """
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
        # Read up to 2 MB looking for a complete JPEG (SOI … EOI)
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
    """Fetch a single JPEG via HTTP GET and decode it."""
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
        # Maybe it's a live MJPEG stream disguised as a plain URL — try parsing it
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
    """
    Non-destructive reachability check.
    Returns '' on success, or a plain-English error string on failure.
    """
    import urllib.request
    import urllib.error
    import socket

    try:
        # Just open the connection and read the first chunk — don't decode.
        req = urllib.request.Request(url, headers={"Connection": "close"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(256)
        return ""
    except (urllib.error.URLError, socket.timeout) as exc:
        return _url_error_message(url, exc)
    except Exception as exc:
        return str(exc)


def _capture_frame(source) -> Image.Image:
    """
    Grab one frame from source (int device index or str URL).

    Auto-detects the URL type:
      • http://…/….jpg  →  single JPEG snapshot (fastest)
      • http://…        →  HTTP MJPEG stream  (parses first JPEG frame)
      • rtsp://…        →  RTSP via cv2.VideoCapture + CAP_FFMPEG
      • int             →  local webcam device index
    """
    import urllib.error, socket

    if isinstance(source, str):
        url_lower = source.lower().split("?")[0]

        # ── RTSP ─────────────────────────────────────────────────────
        if url_lower.startswith("rtsp://"):
            # Try backends in order — conda OpenCV may lack FFmpeg RTSP support
            for backend in (cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY):
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    break
                cap.release()
            else:
                # Last resort: convert rtsp:// → http:// and try MJPEG parser
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

        # ── HTTP snapshot or MJPEG ────────────────────────────────────
        if url_lower.startswith("http://") or url_lower.startswith("https://"):
            if url_lower.endswith((".jpg", ".jpeg")):
                return _fetch_snapshot(source)
            # Generic HTTP — try MJPEG parser first (works for /live, /video, etc.)
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

    # ── Local device ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open local camera device {source}.")
    try:
        cap.read()  # discard first frame (auto-exposure)
        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Camera {source} opened but no frame received.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


class _CameraDialog(tk.Toplevel):
    """
    Full-size camera picker with two panels:
      LEFT  — local / Continuity Camera (auto-scanned listbox)
      RIGHT — iPhone IP / RTSP stream (URL entry)
    """

    _W, _H = 580, 370

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("Select Camera Source")
        self.geometry(f"{self._W}x{self._H}")
        self.resizable(False, False)
        self.grab_set()
        self.configure(bg=BG)
        self.result = None

        self._mode     = tk.StringVar(value="local")   # "local" | "ip"
        self._idx_var  = tk.IntVar(value=config.DEFAULT_CAMERA_INDEX)
        self._url_var  = tk.StringVar(value=config.DEFAULT_STREAM_URL)
        self._cameras: list[int] = []

        self._build()
        self._center(parent)
        threading.Thread(target=self._do_scan, daemon=True).start()

    # ------------------------------------------------------------------
    def _build(self):
        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=TEXT_DARK)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Select Camera Source",
                 font=("Helvetica", 14, "bold"),
                 bg=TEXT_DARK, fg="white", pady=12
        ).pack()

        # ── Tab bar ───────────────────────────────────────────────────
        tab_bar = tk.Frame(self, bg=ACCENT_LINE)
        tab_bar.pack(fill="x")

        self._tab_local = tk.Label(
            tab_bar, text="  Local / Continuity Camera  ",
            font=("Helvetica", 11, "bold"),
            bg=BTN_BLUE, fg="white", pady=8, cursor="hand2")
        self._tab_local.pack(side="left")
        self._tab_local.bind("<Button-1>", lambda _: self._switch("local"))

        self._tab_ip = tk.Label(
            tab_bar, text="  iPhone over WiFi (IP)  ",
            font=("Helvetica", 11, "bold"),
            bg=ACCENT_LINE, fg=TEXT_MID, pady=8, cursor="hand2")
        self._tab_ip.pack(side="left")
        self._tab_ip.bind("<Button-1>", lambda _: self._switch("ip"))

        # ── Content area ──────────────────────────────────────────────
        self._content = tk.Frame(self, bg=BG)
        self._content.pack(fill="both", expand=True, padx=24, pady=16)

        self._panel_local = self._build_local_panel(self._content)
        self._panel_ip    = self._build_ip_panel(self._content)
        self._panel_local.pack(fill="both", expand=True)

        # ── Bottom bar ────────────────────────────────────────────────
        bar = tk.Frame(self, bg="#E8EAF0", pady=10)
        bar.pack(fill="x", side="bottom")
        _ColorButton(bar, "  Cancel  ", self._cancel, BTN_GREY,
                     padx=16, pady=8).pack(side="right", padx=(8, 20))
        _ColorButton(bar, "  Capture  ", self._confirm, BTN_GREEN,
                     font=("Helvetica", 12, "bold"), padx=16, pady=8
        ).pack(side="right", padx=4)

    # ── Local panel ───────────────────────────────────────────────────
    def _build_local_panel(self, parent):
        f = tk.Frame(parent, bg=BG)

        tk.Label(f, text="Detected cameras on this device:",
                 font=("Helvetica", 11), bg=BG, fg=TEXT_DARK
        ).pack(anchor="w", pady=(0, 6))

        # Listbox + scrollbar
        list_frame = tk.Frame(f, bg=ACCENT_LINE, bd=1, relief="flat")
        list_frame.pack(fill="both", expand=True)

        self._listbox = tk.Listbox(
            list_frame,
            font=("Helvetica", 12),
            bg=CARD_BG, fg=TEXT_DARK,
            selectbackground=BTN_BLUE, selectforeground="white",
            activestyle="none",
            relief="flat", bd=0,
            height=5,
        )
        self._listbox.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        sb = tk.Scrollbar(list_frame, orient="vertical",
                          command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.config(yscrollcommand=sb.set)

        self._listbox.insert("end", "  Scanning for cameras…")
        self._listbox.config(state="disabled")

        tk.Label(f,
                 text="Mac tip: iPhone appears automatically as a Continuity Camera\n"
                      "(index 1 or 2) when connected via USB or same WiFi — no app needed.",
                 font=("Helvetica", 9), bg=BG, fg=TEXT_LIGHT, justify="left"
        ).pack(anchor="w", pady=(8, 0))

        return f

    # ── IP panel ──────────────────────────────────────────────────────
    def _build_ip_panel(self, parent):
        f = tk.Frame(parent, bg=BG)

        tk.Label(f, text="Stream URL from your iPhone camera app:",
                 font=("Helvetica", 11), bg=BG, fg=TEXT_DARK
        ).pack(anchor="w", pady=(0, 6))

        entry_frame = tk.Frame(f, bg=ACCENT_LINE, bd=1, relief="flat")
        entry_frame.pack(fill="x")

        self._url_entry = tk.Entry(
            entry_frame, textvariable=self._url_var,
            font=("Helvetica", 12), relief="flat", bd=6,
            bg=CARD_BG, fg=TEXT_DARK, insertbackground=TEXT_DARK,
        )
        self._url_entry.pack(fill="x")

        # Format hint
        tk.Label(f,
                 text="Supported formats (app detects automatically):\n"
                      "  http://192.168.x.x:8080/live       HTTP MJPEG stream\n"
                      "  http://192.168.x.x:8080/shot.jpg   single snapshot\n"
                      "  rtsp://192.168.x.x:554/stream      RTSP",
                 font=("Courier", 9), bg=BG, fg=TEXT_MID, justify="left",
        ).pack(anchor="w", pady=(8, 4))

        # Test connection button + result label
        test_row = tk.Frame(f, bg=BG)
        test_row.pack(fill="x", pady=(10, 0))

        self._test_result = tk.Label(
            test_row, text="", font=("Helvetica", 10),
            bg=BG, fg=TEXT_MID, justify="left", wraplength=460)
        self._test_result.pack(side="right", expand=True, fill="x", padx=(8, 0))

        _ColorButton(test_row, "Test Connection", self._test_connection,
                     BTN_BLUE, padx=12, pady=6,
                     font=("Helvetica", 10, "bold")
        ).pack(side="left")

        tk.Label(f,
                 text="Replace 192.168.x.x with the IP shown in the app.\n"
                      "Works on both Mac and Raspberry Pi over the same WiFi.",
                 font=("Helvetica", 9), bg=BG, fg=TEXT_LIGHT, justify="left"
        ).pack(anchor="w", pady=(8, 0))

        return f

    # ------------------------------------------------------------------
    def _switch(self, mode: str):
        self._mode.set(mode)
        if mode == "local":
            self._panel_ip.pack_forget()
            self._panel_local.pack(fill="both", expand=True)
            self._tab_local.config(bg=BTN_BLUE, fg="white")
            self._tab_ip.config(bg=ACCENT_LINE, fg=TEXT_MID)
        else:
            self._panel_local.pack_forget()
            self._panel_ip.pack(fill="both", expand=True)
            self._tab_ip.config(bg=BTN_BLUE, fg="white")
            self._tab_local.config(bg=ACCENT_LINE, fg=TEXT_MID)
            # Select all only if it still contains the placeholder,
            # so the user can just start typing their IP immediately.
            self._url_entry.focus_set()
            if "192.168.x.x" in self._url_var.get():
                self._url_entry.after(50, lambda: self._url_entry.select_range(0, "end"))

    def _do_scan(self):
        cameras = _scan_cameras()
        self._cameras = cameras
        self.after(0, self._update_listbox)

    def _update_listbox(self):
        self._listbox.config(state="normal")
        self._listbox.delete(0, "end")

        if not self._cameras:
            self._listbox.insert("end", "  No cameras detected")
            self._listbox.config(state="disabled")
            return

        NAMES = {0: "Camera 0  —  built-in / default webcam"}
        for i in self._cameras:
            label = NAMES.get(i, f"Camera {i}  —  Continuity Camera / external")
            self._listbox.insert("end", f"  {label}")

        self._listbox.selection_set(0)
        self._listbox.activate(0)

    def _test_connection(self):
        url = self._url_var.get().strip()
        if not url or "192.168.x.x" in url:
            self._test_result.config(
                text="Enter a real IP address first.", fg="#C0392B")
            return
        self._test_result.config(text="Testing…", fg=TEXT_MID)
        self.update_idletasks()

        def _worker():
            err = _test_connection(url)
            def _done():
                if err:
                    self._test_result.config(text=f"✗  {err.splitlines()[0]}", fg="#C0392B")
                else:
                    self._test_result.config(text="✓  Connected — ready to capture!", fg="#1E8C45")
            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def _confirm(self):
        if self._mode.get() == "ip":
            url = self._url_var.get().strip()
            if not url or "192.168.x.x" in url:
                messagebox.showwarning("Missing URL",
                    "Please enter the actual IP address shown in your iPhone app.",
                    parent=self)
                return
            self.result = url
        else:
            sel = self._listbox.curselection()
            if not sel or not self._cameras:
                messagebox.showwarning("No Camera",
                    "No camera selected. Wait for scanning to finish.",
                    parent=self)
                return
            self.result = self._cameras[sel[0]]
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    def _center(self, parent: tk.Tk):
        px = parent.winfo_x() + parent.winfo_width()  // 2
        py = parent.winfo_y() + parent.winfo_height() // 2
        self.geometry(f"{self._W}x{self._H}+{px - self._W//2}+{py - self._H//2}")


# ==================================================================
# _ColorButton — Label-based button that renders correctly on macOS
# (tk.Button ignores bg/fg on Aqua; tk.Label always respects them)
# ==================================================================

class _ColorButton(tk.Frame):
    """A clickable Label that looks and behaves like a coloured button."""

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

    # --- public interface matches tk.Button.config(state=...) ---------
    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self._lbl.config(bg=self._bg, fg=self._fg, cursor="hand2")
            self.config(bg=self._bg, cursor="hand2")
        else:
            disabled_bg = "#A0A8B4"
            self._lbl.config(bg=disabled_bg, fg="#E0E0E0", cursor="")
            self.config(bg=disabled_bg, cursor="")

    # --- event handlers -----------------------------------------------
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


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    root = tk.Tk()
    SkinClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
