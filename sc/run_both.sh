#!/bin/bash
# Headless dataset recording with automatic Sentire boot
#
# Usage:
#   ./run_both.sh
#
# This script:
# 1. Boots a JACK server with pw-jack
# 2. Runs sclang with the combined driver script
# 3. The driver loads Sentire, waits for it, then starts recording
# 4. Exits automatically when recording is complete

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER_SCRIPT="$SCRIPT_DIR/run_both.scd"

echo "==================================="
echo "Dataset Recording - Headless Mode"
echo "==================================="
echo ""
echo "Driver: $DRIVER_SCRIPT"
echo ""
echo "Starting sclang with pw-jack..."
echo "Watch the output for progress."
echo ""
echo "The recording will:"
echo "  1. Boot Sentire headless"
echo "  2. Wait for initialization"
echo "  3. Start recording dataset"
echo "  4. Exit when complete"
echo ""
echo "To stop early, press Ctrl+C"
echo ""

pw-jack sclang "$DRIVER_SCRIPT"
