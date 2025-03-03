import cv2
import numpy as np
import glob

CHECKERBOARD = (8,5)  # 8x5 inner corners (9x6 squares)
SQUARE_SIZE = 0.025  # Physical size of a square in meters

# Prepare object points (3D coordinates)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points
img_size = None  # Initialize image size

images = glob.glob("./calibrationImgs//calibration_shot_*.jpg")

if not images:
    raise ValueError("No calibration images found in the specified directory.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Invalid image: {fname}")
        continue  # Skip invalid images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
    )
    
    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners_refined)
    else:
        print(f"No chessboard found in {fname}")

# Check if valid images were processed
if len(objpoints) == 0:
    raise ValueError("No valid calibration images with detected chessboards!")

img_size = (gray.shape[1], gray.shape[0])  # Set image size after processing

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)

print(f"Successfully processed {len(objpoints)} images.")
# Save calibration data
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)

print("Calibration complete!")
