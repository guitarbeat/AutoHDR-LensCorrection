#!/bin/bash
set -e

# Automatically exit if the kaggle command isn't configured
command -v kaggle >/dev/null 2>&1 || { echo >&2 "Kaggle CLI is not installed or not in PATH. Exiting."; exit 1; }

DATA_ROOT="${AUTOHDR_DATA_ROOT:-/Volumes/Love SSD}"

echo "Navigating to ${DATA_ROOT}..."
mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo "Downloading the Kaggle dataset. This is a 37GB file and may take a few hours depending on your internet connection."
kaggle competitions download -c automatic-lens-correction

echo "Download complete! Now extracting the dataset..."
unzip -q automatic-lens-correction.zip

echo "Extraction finished. Cleaning up the zip file to save disk space..."
rm automatic-lens-correction.zip

echo "All done! Data is available in ${DATA_ROOT}"
