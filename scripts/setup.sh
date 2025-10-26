#!/bin/bash
# Setup script for SecIDS-v2

echo "=================================="
echo "SecIDS-v2 Setup"
echo "=================================="

# Check Python version
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || { echo "Warning: Could not activate venv, installing globally"; }

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Generate data: python scripts/generate_synthetic_data.py"
echo "  2. Train model: python -m training.train --data data/train.parquet"
echo ""
