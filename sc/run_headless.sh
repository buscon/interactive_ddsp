#!/bin/bash
# Run dataset recording in headless mode
#
# Usage:
#   ./run_headless.sh
#
# This script:
# 1. Boots a JACK server with pw-jack
# 2. Runs sclang with the standalone recording script
# 3. The script auto-loads Sentire and starts recording

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDING_SCRIPT="$SCRIPT_DIR/record_dataset_standalone.scd"

echo "==================================="
echo "Dataset Recording - Headless Mode"
echo "==================================="
echo ""
echo "Script: $RECORDING_SCRIPT"
echo ""
echo "Starting sclang with pw-jack..."
echo "Watch the output for progress."
echo ""
echo "To stop early, press Ctrl+C"
echo ""

pw-jack sclang "$RECORDING_SCRIPT"
