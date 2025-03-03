import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from picamera2 import Picamera2
import cv2
import numpy as np
from pupil_apriltags import Detector

# Load camera calibration data (if available)
try:
    camera_matrix = np.load("camera_matrix.npy")
    dist_coeffs = np.load("dist_coeffs.npy")
except FileNotFoundError:
    print("Calibration files not found. Using default values.")
    camera_matrix = np.eye(3)  # Placeholder identity matrix
    dist_coeffs = np.zeros((5, 1))  # No distortion

# Initialize the AprilTag detector (using 'tag36h11' family)
detector = Detector(families='tag36h11')

# Define the physical size of the AprilTag (in meters)
TAG_SIZE = 0.1  # 10 cm

# Initialize and configure the Raspberry Pi camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1920,1080)})
picam2.configure(config)
picam2.start()

# Set up Matplotlib for live display
fig, ax = plt.subplots()
ax.axis('off')  # Hide axes for a cleaner display
initial_frame = picam2.capture_array()  # Capture an initial frame
img = ax.imshow(initial_frame)  # Create an image object for updating

# Function to draw 3D axes on the detected tag
def draw_axis(frame, rvec, tvec):
    axis_points = np.float32([
        [0, 0, 0],           # Origin
        [TAG_SIZE, 0, 0],    # X-axis
        [0, TAG_SIZE, 0],    # Y-axis
        [0, 0, -TAG_SIZE]    # Z-axis
    ])
    # Project 3D points to 2D image plane
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.astype(int)  # Convert to integers
    h, w = frame.shape[:2]  # Get image dimensions
    origin = tuple(img_points[0].ravel())  # Starting point of axes
    # Draw each axis with bounds checking
    for i, color in zip([1, 2, 3], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        pt = img_points[i].ravel()
        if 0 <= pt[0] < w and 0 <= pt[1] < h:  # Check if point is within image
            cv2.line(frame, origin, tuple(pt), color, 3)
        else:
            print(f"Warning: Axis point {i} out of bounds: {pt}")

# Define the update function for the animation
def update(frame_number):
    frame = picam2.capture_array()  # Capture a new frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    # Detect AprilTags
    tags = detector.detect(gray)
    for tag in tags:
        # Define 3D object points for the tag
        object_pts = np.array([
            [-TAG_SIZE/2, -TAG_SIZE/2, 0],
            [TAG_SIZE/2, -TAG_SIZE/2, 0],
            [TAG_SIZE/2, TAG_SIZE/2, 0],
            [-TAG_SIZE/2, TAG_SIZE/2, 0]
        ], dtype=np.float32)

        # Solve for pose using calibration data
        success, rvec, tvec = cv2.solvePnP(
            object_pts,
            tag.corners.astype(np.float32),
            camera_matrix,
            dist_coeffs
        )

        if success:
            # Check for unreasonable translation values (e.g., >10 meters)
            if np.any(np.abs(tvec) > 10):  # Threshold in meters
                print(f"Warning: Unreasonable pose for Tag {tag.tag_id}: tvec={tvec.flatten()}")
                continue
            x, y, z = tvec.flatten() * 100  # Convert to cm
            cv2.polylines(frame, [tag.corners.astype(int)], True, (0, 255, 255), 2)
            draw_axis(frame, rvec, tvec)
            # Display coordinates on the frame
            cv2.putText(frame, f"X: {x:.1f}cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Y: {y:.1f}cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Z: {z:.1f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Detected Tag {tag.tag_id} at X:{x:.1f}cm Y:{y:.1f}cm Z:{z:.1f}cm")

    # Update the Matplotlib image with the processed frame
    img.set_data(frame)
    return img,

# Create the animation
ani = FuncAnimation(fig, update, interval=33, cache_frame_data=False)

# Display the plot
plt.show()

# Stop the camera when the window is closed
picam2.stop()
