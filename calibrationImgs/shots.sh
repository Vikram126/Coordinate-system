#!/bin/bash

# Number of images to capture
NUM_IMAGES=20
sleep 1
# Loop to capture images with sequential numbering
for ((i=1; i<=NUM_IMAGES; i++)); do
    FILENAME=$(printf "calibration_shot_%02d.jpg" "$i")
    libcamera-still -o "$FILENAME"
    echo "Captured: $FILENAME"
    sleep 1  # Optional: Adds a delay between captures
done

echo "All images captured."
