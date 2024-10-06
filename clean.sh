#!/bin/bash

# Remove all __pycache__ directories
find src/ -type d -name "__pycache__" -exec rm -rf {} +

# Remove all tmp directories
find src/ -type d -name "tmp" -exec rm -rf {} +

# Remove all .log files in the current directory and its subdirectories
find . -type f -name "*.log" -exec rm -f {} +

# Remove all files in the output directory, but keep the directory itself
find ./output/ -type f -exec rm -rf {} +

echo "All __pycache__, tmp directories, .log files, and contents of the output folder have been removed."
