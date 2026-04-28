#!/usr/bin/env bash
# Creates a virtual environment and installs all DPO pipeline dependencies.
set -e

VENV_DIR="$(dirname "$0")/venv"

# Prefer python3.10+ when available (required by several dependencies).
PYTHON=$(command -v python3.10 || command -v python3.11 || command -v python3.12 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version))"

echo "Creating virtual environment at $VENV_DIR ..."
"$PYTHON" -m venv "$VENV_DIR"

echo "Activating and installing packages ..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

echo "Installing project-specific extras ..."
pip install "evalplus==0.3.1" wandb human-eval

# bitsandbytes is only needed for --use_4bit on Linux+CUDA; skip on macOS.
if [[ "$(uname)" != "Darwin" ]]; then
    pip install "bitsandbytes>=0.43.0"
fi

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
