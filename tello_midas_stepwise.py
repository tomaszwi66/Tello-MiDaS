"""
Tello-MiDaS Navigator - Stepwise (v1)
======================================
The first working version of the indoor navigation system.
The drone moves in discrete steps: analyze -> rotate -> move -> stop.
Flight is safe but jerky - like a frog jumping between positions.

This "stop-and-go" behavior comes from using discrete Tello commands:
  - tello.rotate_clockwise(N)  -> drone stops, rotates N degrees, stops
  - tello.move_forward(N)      -> drone stops, moves N cm, stops

The continuous version (tello_midas_continuous.py) replaces these with
send_rc_control() to blend rotation and forward motion simultaneously,
resulting in smooth, fluid flight.

Core logic:
  - "Fly Forward" rule: find the darkest region (furthest space), fly toward it
  - "Don't Crash" rule: detect bright (close) regions on the sides, rotate away

Note on close_threshold = 1000:
  cv2.minMaxLoc on a uint8 depth map returns values in range [0, 255],
  so this threshold is never triggered in practice - side obstacle avoidance
  is effectively disabled. This turned out to work better empirically:
  aggressive side-avoidance caused more problems than it solved.
  The "fly to the furthest point" rule does most of the heavy lifting alone.

Known failure modes:
  - Corner Trap: MiDaS reads room corners as deep open space -> drone flies into the wall
  - Invisible Glass: textureless white walls / windows -> no reliable depth signal
  - No localization: no SLAM, no odometry, purely reactive behavior

Author: Tomasz Wietrzykowski
  Portfolio: https://tomaszwi66.github.io/
  LinkedIn:  https://www.linkedin.com/in/tomasz-wietrzykowski

Hardware: DJI Tello + old VAIO laptop (CPU only)
Tested: November 2023
"""

import cv2
import torch
import numpy as np
import time
from djitellopy import Tello

# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Load MiDaS ────────────────────────────────────────────────────────────────
# MiDaS_small: fast enough for real-time on older hardware.
# Produces *relative* depth - values are not metric. Bright = close, dark = far.
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas = midas.to(device)
midas.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# ── Connect to Tello ──────────────────────────────────────────────────────────
tello = Tello()
tello.connect()
tello.streamon()

# Lock altitude to a fixed flight plane.
# The Tello camera faces slightly downward, so without this the drone
# would interpret the floor as open space and climb into the ceiling.
tello.takeoff()
try:
    tello.move_up(30)  # Adjust the height as needed
except Exception as e:
    print(f"Error moving up: {e}")

# ── Display window ─────────────────────────────────────────────────────────────
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Depth Map', 360, 240)

# ── Navigation constants ───────────────────────────────────────────────────────
DEPTH_WIDTH, DEPTH_HEIGHT = 360, 240
CENTER_DEADZONE = DEPTH_WIDTH // 20       # pixels - ignore small offsets from center
ROTATION_SPEED_MIN = 10                   # degrees min yaw command
ROTATION_SPEED_MAX = 60                   # degrees max yaw command
DEPTH_DARK_BAND = 30                      # threshold band above min for "far" region detection
FORWARD_STEP_CM = 20                      # cm to move forward per step
ROTATION_SLEEP = 0.5                      # seconds to wait after rotate command

# Note: value > 255 means side obstacle avoidance is effectively disabled.
# See docstring for explanation - this was intentional after testing.
CLOSE_THRESHOLD = 1000

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    try:
        # Get frame from Tello video stream
        frame = tello.get_frame_read().frame

        # Transform data for the MiDaS model
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgbatch = transform(img).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

            output = prediction.cpu().numpy()

            # Normalize depth values to the range 0-255
            normalized_depth = (output - output.min()) / (output.max() - output.min()) * 255
            depth_map = np.uint8(normalized_depth)

            # Resize depth map to 360x240
            depth_map_resized = cv2.resize(depth_map, (DEPTH_WIDTH, DEPTH_HEIGHT))

            # Apply threshold to find the darkest area (furthest open space)
            _, thresholded = cv2.threshold(depth_map_resized, np.min(depth_map_resized) + DEPTH_DARK_BAND, 255, cv2.THRESH_BINARY_INV)

            # Find contours of the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            should_move_forward = False
            if contours:
                # Assuming the largest contour is the farthest area
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(depth_map_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
                center_of_farthest_area = (x + w//2, y + h//2)
                cv2.circle(depth_map_resized, center_of_farthest_area, 5, (255, 0, 0), -1)

                # Calculate the difference between the center of the frame and the farthest point
                frame_center = (depth_map_resized.shape[1] // 2, depth_map_resized.shape[0] // 2)
                diff = (center_of_farthest_area[0] - frame_center[0], center_of_farthest_area[1] - frame_center[1])

                # Rotate drone towards the farthest area's center (discrete command - drone stops to rotate)
                if abs(diff[0]) > CENTER_DEADZONE:
                    rotation_speed = max(ROTATION_SPEED_MIN, min(ROTATION_SPEED_MAX, abs(diff[0]) // 10))
                    if diff[0] > 0:
                        tello.rotate_clockwise(rotation_speed)
                    else:
                        tello.rotate_counter_clockwise(rotation_speed)
                    time.sleep(ROTATION_SLEEP)
                else:
                    should_move_forward = True

            # Split the depth map into left and right regions to detect very close objects
            left_region = depth_map_resized[:, :depth_map_resized.shape[1] // 2]
            right_region = depth_map_resized[:, depth_map_resized.shape[1] // 2:]

            # Find the brightest area (closest object) in both regions
            _, max_val_left, _, _ = cv2.minMaxLoc(left_region)
            _, max_val_right, _, _ = cv2.minMaxLoc(right_region)

            # Threshold for detecting very close objects
            # Set to 1000 (above uint8 range) - effectively disables side avoidance.
            # Empirically this worked better: see docstring for explanation.
            close_threshold = CLOSE_THRESHOLD

            if max_val_left > close_threshold and max_val_right > close_threshold:
                should_move_forward = False
                print("Close objects detected on both sides. Staying put.")
            elif max_val_left > close_threshold:
                tello.rotate_clockwise(30)
                print("Close object detected on the left, rotating right.")
                time.sleep(1)
                should_move_forward = False
            elif max_val_right > close_threshold:
                tello.rotate_counter_clockwise(30)
                print("Close object detected on the right, rotating left.")
                time.sleep(1)
                should_move_forward = False

            # Move forward in a discrete step (drone stops after each move)
            if should_move_forward:
                print("Moving forward")
                tello.move_forward(FORWARD_STEP_CM)

            # Display depth map using OpenCV
            cv2.imshow('Depth Map', depth_map_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        tello.land()
        break

# ── Cleanup ────────────────────────────────────────────────────────────────────
tello.land()
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
