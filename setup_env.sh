#!/usr/bin/env bash
# Creates a virtual environment and installs all DPO pipeline dependencies.
set -e

VENV_DIR="$(dirname "$0")/venv"

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Activating and installing packages ..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
