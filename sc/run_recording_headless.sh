#!/bin/bash
# Run the dataset recording in headless mode
# 
# Usage:
#   ./run_recording_headless.sh [path-to-sentire]
#
# Example:
#   ./run_recording_headless.sh ~/sentire

SENTIRE_FOLDER="${1:-$HOME/sentire}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SENTIRE_FOLDER" ]; then
    echo "ERROR: Sentire folder not found: $SENTIRE_FOLDER"
    echo "Usage: $0 [path-to-sentire]"
    echo "Example: $0 ~/sentire"
    exit 1
fi

echo "Loading Sentire from: $SENTIRE_FOLDER"
echo "Recording script: $SCRIPT_DIR/record_dataset.scd"
echo ""
echo "Starting sclang in headless mode..."
echo "Check the SuperCollider post window for progress."
echo ""

# Run sclang with both scripts
sclang -scdefs "$SENTIRE_FOLDER/main-headless.scd,$SCRIPT_DIR/record_dataset.scd"
