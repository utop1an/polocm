#!/bin/bash

# Remove all __pycache__ directories
find src/ -type d -name "__pycache__" -exec rm -rf {} +

# Remove all tmp directories
find src/ -type d -name "tmp" -exec rm -rf {} +

echo "All __pycache__ and tmp directories removed from src/"
