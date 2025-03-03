# AprilTag Detection and Pose Estimation with Raspberry Pi Camera

This project captures live video from a Raspberry Pi camera, detects AprilTags (specifically from the `tag36h11` family), estimates their 3D poses using camera calibration data, and visualizes the results in real-time. The visualization includes the tag boundaries and 3D axes (X, Y, Z) overlaid on the video feed, displayed using `matplotlib`.

## Project Overview
- **Camera Capture**: Uses `picamera2` to capture frames from a Raspberry Pi camera.
- **AprilTag Detection**: Detects AprilTags using the `pupil-apriltags` library.
- **Pose Estimation**: Estimates the 3D pose (position and orientation) of each detected tag using `cv2.solvePnP`.
- **Visualization**: Draws the tag boundaries and 3D axes on the video feed and displays the tag’s position (X, Y, Z in centimeters).
- **Real-Time Display**: Uses `matplotlib` for live video display and updates.

---

## Prerequisites

### Hardware
- Raspberry Pi (tested on Raspberry Pi 5 with Bullseye OS)
- Raspberry Pi Camera Module (compatible with your Raspberry Pi model)
- AprilTag (from the `tag36h11` family, size 10 cm)

### Software
- Python 3.7 or later
- Required Python libraries:
  ```bash
  pip install pupil-apriltags opencv-python-headless matplotlib picamera2
