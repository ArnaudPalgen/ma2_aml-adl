#!/usr/bin/env bash

DATASET="kinetics400"
DATA_DIR="../../../data/${DATASET}"

echo "Removing corrupt files..."

find ${DATA_DIR} -type f -name '*.mp4' -exec bash is_corrupt.sh {} \;

echo "Corrupt files removed"