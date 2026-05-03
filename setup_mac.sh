#!/usr/bin/env bash
# setup_mac.sh — One-shot setup for Skin Disease Classifier on macOS
#
# Fixes handled automatically:
#   • PyPI opencv-python wheels crash on macOS 26 (Tahoe) with
#     "macOS 26 (2603) or later required" — we use conda-forge opencv
#     instead, which compiles for the exact Python version in the env
#     and contains no macOS marketing-version check code.
#   • Homebrew opencv is compiled for Python 3.14 and cannot be
#     symlinked into a Python 3.10 env — conda-forge sidesteps this.
#
# Usage:
#   bash setup_mac.sh
set -euo pipefail

CONDA_ENV="skinsential"
PYTHON_VER="3.10"

# ── Colours ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

echo -e "\n${BOLD}Skin Disease Classifier — Mac Setup${NC}\n"
echo    "  Python  : ${PYTHON_VER}"
echo    "  opencv  : conda-forge (avoids macOS Tahoe PyPI crash)"
echo    "  env     : ${CONDA_ENV}"
echo ""

# ── 1. Homebrew (needed only for Miniforge + python-tk) ───────────────
info "Checking Homebrew..."
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
# Ensure brew is on PATH (Apple Silicon)
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || true
eval "$(brew shellenv 2>/dev/null)" || true
success "Homebrew $(brew --version | head -1)"

# ── 2. python-tk (required for tkinter — must be on system, not conda) ─
info "Checking python-tk@${PYTHON_VER}..."
if ! brew list "python-tk@${PYTHON_VER}" &>/dev/null; then
    info "Installing python-tk@${PYTHON_VER}..."
    brew install "python-tk@${PYTHON_VER}"
fi
success "python-tk@${PYTHON_VER} ready"

# ── 3. Miniforge ──────────────────────────────────────────────────────
info "Checking Miniforge / conda..."
_source_conda() {
    local base
    base=$(conda info --base 2>/dev/null || echo "")
    [[ -n "$base" ]] && source "${base}/etc/profile.d/conda.sh" 2>/dev/null || true
}

if command -v conda &>/dev/null; then
    _source_conda
    success "conda $(conda --version) already installed"
else
    info "Installing Miniforge via Homebrew..."
    brew install miniforge
    CONDA_BASE="$(brew --prefix)/Caskroom/miniforge/base"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda init zsh  2>/dev/null || true
    conda init bash 2>/dev/null || true
    success "Miniforge installed — shell init done for zsh + bash"
fi

# ── 4. Conda environment ──────────────────────────────────────────────
info "Checking conda environment '${CONDA_ENV}'..."
if conda env list | grep -q "^${CONDA_ENV}[[:space:]]"; then
    warn "Environment '${CONDA_ENV}' already exists — skipping creation."
    warn "Delete and recreate with:  conda env remove -n ${CONDA_ENV}"
else
    info "Creating '${CONDA_ENV}' with Python ${PYTHON_VER}..."
    conda create -n "${CONDA_ENV}" python="${PYTHON_VER}" -y
fi
conda activate "${CONDA_ENV}"
success "Activated: $(python --version)"

# ── 5. opencv + tk via conda-forge ───────────────────────────────────
# conda-forge builds opencv against the exact Python ABI in the env.
# No PyPI wheels, no macOS marketing-version check — works on Tahoe.
info "Installing opencv and tk from conda-forge..."
conda install -c conda-forge opencv tk -y --quiet
success "opencv $(python -c 'import cv2; print(cv2.__version__)') installed"

# ── 6. Python packages via pip ────────────────────────────────────────
info "Installing tensorflow, numpy, Pillow, matplotlib..."
pip install --upgrade pip --quiet
pip install "tensorflow>=2.12.0,<2.17.0" numpy Pillow matplotlib --quiet
success "Python packages installed"

# ── 7. Verify all imports ─────────────────────────────────────────────
info "Verifying imports..."
FAILED=0
check() {
    local label="$1"; shift
    if python -c "$@" 2>/dev/null; then
        : # success printed inline
    else
        warn "${label} import FAILED"
        FAILED=1
    fi
}

check "cv2"        "import cv2;         print(f'  cv2        {cv2.__version__}')"
check "tensorflow" "import tensorflow as tf; print(f'  tensorflow {tf.__version__}')"
check "tkinter"    "import tkinter;     print(f'  tkinter    OK')"
check "Pillow"     "import PIL;         print(f'  Pillow     {PIL.__version__}')"
check "numpy"      "import numpy as np; print(f'  numpy      {np.__version__}')"
check "matplotlib" "import matplotlib;  print(f'  matplotlib {matplotlib.__version__}')"

if [[ $FAILED -eq 1 ]]; then
    echo ""
    warn "One or more imports failed. Check messages above."
    warn "Try running the script again — transient network errors can cause failures."
    exit 1
fi

success "All imports OK"

# ── 8. Done ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Setup complete!${NC}"
echo ""
echo "To run the app (open a new terminal tab first if conda wasn't active before):"
echo ""
echo -e "  ${CYAN}conda activate ${CONDA_ENV}${NC}"
echo -e "  ${CYAN}cd $(pwd)${NC}"
echo -e "  ${CYAN}python app.py${NC}"
echo ""
echo "If 'conda activate' fails in a new terminal, restart the terminal"
echo "once so the shell init (added to ~/.zshrc) takes effect."
echo ""
