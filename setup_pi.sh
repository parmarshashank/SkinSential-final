#!/usr/bin/env bash
# setup_pi.sh — One-shot setup for Skin Disease Classifier on Raspberry Pi 4
# Usage: bash setup_pi.sh
#
# Installs: Miniforge → conda env (Python 3.10) → opencv + tk + tensorflow
set -euo pipefail

CONDA_ENV="skinsential"
PYTHON_VER="3.10"
MINIFORGE_INSTALLER="Miniforge3-Linux-aarch64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/${MINIFORGE_INSTALLER}"

# ── Colours ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo -e "\n${BOLD}Skin Disease Classifier — Raspberry Pi 4 Setup${NC}\n"

# ── Verify ARM64 ──────────────────────────────────────────────────────
[[ "$(uname -m)" == "aarch64" ]] || die "This script is for Raspberry Pi (aarch64). Got: $(uname -m)"

# ── 1. System packages ────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq
# libatlas-base-dev removed from Pi OS Bookworm — libopenblas-dev replaces it.
sudo apt-get install -y \
    libopenblas-dev libhdf5-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    python3-tk wget curl git \
    > /dev/null
success "System packages ready"

# ── 2. Miniforge ──────────────────────────────────────────────────────
info "Checking Miniforge / conda..."

# Find or install conda, then expose the conda shell function for this script.
_init_conda() {
    # Try common install locations
    for base in "$HOME/miniforge3" "$HOME/anaconda3" "$HOME/miniconda3" \
                "/opt/conda" "/opt/miniforge3"; do
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            source "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    done
    # Last resort: ask conda itself
    local cb
    cb=$(conda info --base 2>/dev/null) && \
        source "${cb}/etc/profile.d/conda.sh" && return 0
    return 1
}

if command -v conda &>/dev/null || _init_conda 2>/dev/null; then
    _init_conda 2>/dev/null || true
    success "conda $(conda --version) already installed"
else
    info "Downloading Miniforge (~80 MB)..."
    wget -q --show-progress "${MINIFORGE_URL}" -O "/tmp/${MINIFORGE_INSTALLER}"
    info "Installing Miniforge to ~/miniforge3..."
    bash "/tmp/${MINIFORGE_INSTALLER}" -b -p "$HOME/miniforge3"
    rm "/tmp/${MINIFORGE_INSTALLER}"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda init bash 2>/dev/null || true
    success "Miniforge installed"
fi

# ── 3. Conda environment ──────────────────────────────────────────────
info "Checking conda environment '${CONDA_ENV}'..."
if conda env list | grep -q "^${CONDA_ENV}[[:space:]]"; then
    warn "Environment '${CONDA_ENV}' already exists — skipping creation."
else
    info "Creating '${CONDA_ENV}' (Python ${PYTHON_VER})..."
    conda create -n "${CONDA_ENV}" python="${PYTHON_VER}" -y
fi
success "Conda environment ready"

# ── Helper: run commands inside the env without needing 'conda activate'
# 'conda activate' requires an interactive shell; 'conda run' works anywhere.
CR() { conda run -n "${CONDA_ENV}" --no-capture-output "$@"; }

# ── 4. opencv + tk via conda-forge ───────────────────────────────────
info "Installing opencv and tk from conda-forge..."
conda install -n "${CONDA_ENV}" -c conda-forge opencv tk -y --quiet
success "opencv $(CR python -c 'import cv2; print(cv2.__version__)') installed"

# ── 5. Python packages via pip ────────────────────────────────────────
# Ensure pip exists inside the env first (not always installed by default).
info "Ensuring pip is installed in the env..."
conda install -n "${CONDA_ENV}" pip -y --quiet
success "pip ready"

info "Upgrading pip..."
CR python -m pip install --upgrade pip --quiet

info "Installing tensorflow, numpy, Pillow, matplotlib..."
CR python -m pip install tensorflow numpy Pillow matplotlib --quiet
success "Python packages installed"

# ── 6. Verify imports ─────────────────────────────────────────────────
info "Verifying imports..."
FAILED=0
check() {
    local label="$1"; shift
    CR python -c "$@" 2>/dev/null || { warn "${label} import FAILED"; FAILED=1; }
}

check "cv2"        "import cv2;              print(f'  cv2        {cv2.__version__}')"
check "tensorflow" "import tensorflow as tf; print(f'  tensorflow {tf.__version__}')"
check "tkinter"    "import tkinter;          print(f'  tkinter    OK')"
check "Pillow"     "import PIL;              print(f'  Pillow     {PIL.__version__}')"
check "numpy"      "import numpy as np;      print(f'  numpy      {np.__version__}')"
check "matplotlib" "import matplotlib;       print(f'  matplotlib {matplotlib.__version__}')"

if [[ $FAILED -eq 1 ]]; then
    warn "Some imports failed — check messages above."
    exit 1
fi
success "All imports OK"

# ── 7. Done ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Setup complete!${NC}"
echo ""
echo "To run the app — open a new terminal (so conda init takes effect), then:"
echo ""
echo -e "  ${CYAN}conda activate ${CONDA_ENV}${NC}"
echo -e "  ${CYAN}cd ~/proj${NC}"
echo -e "  ${CYAN}python app.py${NC}"
echo ""
echo "Note: First launch imports TensorFlow — expect ~15 sec startup on Pi."
echo ""
