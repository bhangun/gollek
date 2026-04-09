#!/bin/bash
# Jupyter Java Kernel for Gollek SDK
# This script installs the Jupyter kernel configuration for gollek-sdk

set -e

KERNEL_NAME="Gollek-SDK"
KERNEL_DIR="$(jupyter --data-dir)/kernels/gollek-sdk"

echo "Installing Gollek SDK Jupyter Kernel..."
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create kernel directory
mkdir -p "$KERNEL_DIR"

# Copy kernel.json
cp "$SCRIPT_DIR/kernel.json" "$KERNEL_DIR/"

echo "✅ Kernel installed to: $KERNEL_DIR"
echo ""
echo "To use the kernel:"
echo "  1. Start Jupyter: jupyter notebook"
echo "  2. Create new notebook: New → Gollek-SDK"
echo ""
echo "Installation complete!"
