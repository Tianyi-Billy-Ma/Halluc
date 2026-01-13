#!/bin/bash
# Script to clean up all __pycache__ directories recursively

# Get the directory where the script is located, or use current directory
TARGET_DIR="${1:-.}"

echo "Searching for __pycache__ directories in: $TARGET_DIR"

# Find and remove all __pycache__ directories
FOUND=$(find "$TARGET_DIR" -type d -name "__pycache__" 2>/dev/null)

if [ -z "$FOUND" ]; then
    echo "No __pycache__ directories found."
else
    echo "Found the following __pycache__ directories:"
    echo "$FOUND"
    echo ""
    echo "Removing..."
    find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "Done! All __pycache__ directories have been removed."
fi

# Also remove .pyc files that might be outside __pycache__
PYC_FILES=$(find "$TARGET_DIR" -type f -name "*.pyc" 2>/dev/null)
if [ -n "$PYC_FILES" ]; then
    echo ""
    echo "Also removing standalone .pyc files..."
    find "$TARGET_DIR" -type f -name "*.pyc" -delete 2>/dev/null
    echo "Done!"
fi
