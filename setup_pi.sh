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

# ── Verify we're on ARM64 ─────────────────────────────────────────────
ARCH=$(uname -m)
if [[ "$ARCH" != "aarch64" ]]; then
    die "This script is for Raspberry Pi (aarch64). Detected: ${ARCH}"
fi

# ── 1. System packages ────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq
# libatlas-base-dev was removed from Pi OS Bookworm — libopenblas-dev is the replacement.
sudo apt-get install -y \
    libopenblas-dev libhdf5-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    python3-tk wget curl git \
    > /dev/null
success "System packages ready"

# ── 2. Miniforge ──────────────────────────────────────────────────────
info "Checking Miniforge / conda..."
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
    success "conda $(conda --version) already installed"
else
    info "Downloading Miniforge installer (~80 MB)..."
    wget -q --show-progress "${MINIFORGE_URL}" -O "/tmp/${MINIFORGE_INSTALLER}"

    info "Installing Miniforge to ~/miniforge3 ..."
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
    warn "To recreate: conda env remove -n ${CONDA_ENV}"
else
    info "Creating conda environment '${CONDA_ENV}' (Python ${PYTHON_VER})..."
    conda create -n "${CONDA_ENV}" python="${PYTHON_VER}" -y
fi
success "Conda environment ready"

# ── 4. Activate and install packages ─────────────────────────────────
info "Activating '${CONDA_ENV}' and installing packages..."
conda activate "${CONDA_ENV}"
pip install --upgrade pip --quiet

# opencv + tk from conda-forge (pre-built ARM64, no compile needed)
info "Installing opencv and tk from conda-forge..."
conda install -c conda-forge opencv tk -y --quiet
success "opencv and tk installed"

# Python packages
info "Installing tensorflow, Pillow, matplotlib, numpy..."
pip install tensorflow numpy Pillow matplotlib --quiet
success "Python packages installed"

# ── 5. Verify imports ─────────────────────────────────────────────────
info "Verifying imports..."
FAILED=0

python -c "import cv2;         print(f'  cv2        {cv2.__version__}')"      || { warn "cv2 failed";        FAILED=1; }
python -c "import tensorflow as tf; print(f'  tensorflow {tf.__version__}')"  || { warn "tensorflow failed"; FAILED=1; }
python -c "import tkinter;     print(f'  tkinter    OK')"                      || { warn "tkinter failed";    FAILED=1; }
python -c "import PIL;         print(f'  Pillow     {PIL.__version__}')"       || { warn "Pillow failed";     FAILED=1; }
python -c "import numpy as np; print(f'  numpy      {np.__version__}')"        || { warn "numpy failed";      FAILED=1; }

if [[ $FAILED -eq 1 ]]; then
    warn "Some imports failed — check warnings above."
else
    success "All imports OK"
fi

# ── 6. Copy reminder ─────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Copy project files to this Pi (from your Mac):"
echo -e "     ${CYAN}scp -r user@mac-ip:~/Desktop/proj ~/proj${NC}"
echo ""
echo "  2. Run the app:"
echo -e "     ${CYAN}conda activate ${CONDA_ENV}${NC}"
echo -e "     ${CYAN}cd ~/proj${NC}"
echo -e "     ${CYAN}python app.py${NC}"
echo ""
echo "Note: First run imports TensorFlow — expect ~15 sec startup on Pi."
echo ""
