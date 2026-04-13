"""
Tello-MiDaS Navigator - v2: Continuous RC Control (the "fluid flight" version)
===============================================================================
Key improvement over v1: instead of discrete rotate -> move -> stop steps,
this version uses send_rc_control() to blend yaw correction and forward motion
simultaneously. The result is smooth, continuous flight rather than frog leaps.

What changed from v1:
  - tello.rotate_clockwise(N) replaced with tello.send_rc_control(lr, fb, ud, yaw)
    This sends raw RC values (range -100 to 100) and allows simultaneous axes.
  - The drone now corrects its heading *while* moving forward.
  - Rotation speed is still proportional to the visual offset from center.
  - Obstacle check extracted into a helper function for clarity.
  - Battery check added on startup.

Note on close_threshold = 800:
  cv2.minMaxLoc on a uint8 depth map returns values in range [0, 255],
  so this threshold is also never triggered in practice - same behavior as v1.
  Side obstacle avoidance is effectively disabled in both versions.
  The "fly to the furthest point" rule handles navigation on its own.

send_rc_control(left_right, forward_back, up_down, yaw)
  All values: -100 to 100
  Positive yaw = clockwise rotation
  Positive forward_back = forward motion
  (0, 0, 0, 0) = hover in place

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
battery_level = tello.get_battery()
print(f"Battery: {battery_level}%")
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
ROTATION_SPEED_MIN = 10                   # RC units min yaw
ROTATION_SPEED_MAX = 60                   # RC units max yaw
FORWARD_SPEED = 20                        # RC units - constant forward speed
DEPTH_DARK_BAND = 30                      # threshold band above min for "far" region detection

# Note: value > 255 means side obstacle avoidance is effectively disabled.
# See docstring for explanation - this was intentional after testing.
CLOSE_THRESHOLD = 800


# ── Helper functions ──────────────────────────────────────────────────────────

def calculate_rotation_speed(diff, frame_width):
    """Proportional yaw speed based on horizontal offset from frame center."""
    return max(ROTATION_SPEED_MIN, min(ROTATION_SPEED_MAX, abs(diff) // 10))

def check_for_obstacles(depth_map_resized, close_threshold=CLOSE_THRESHOLD):
    """
    Split the depth map into left and right halves.
    Return the maximum brightness value in each half.
    Bright pixels = close objects.
    """
    left_region = depth_map_resized[:, :depth_map_resized.shape[1] // 2]
    right_region = depth_map_resized[:, depth_map_resized.shape[1] // 2:]
    _, max_val_left, _, _ = cv2.minMaxLoc(left_region)
    _, max_val_right, _, _ = cv2.minMaxLoc(right_region)
    return max_val_left, max_val_right


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

                # Key improvement vs v1: blend yaw correction with forward motion simultaneously.
                # Rotation speed is proportional to how far the target is from center.
                if abs(diff[0]) > CENTER_DEADZONE:
                    rotation_speed = calculate_rotation_speed(diff[0], depth_map_resized.shape[1])
                    if diff[0] > 0:
                        tello.send_rc_control(0, FORWARD_SPEED, 0, rotation_speed)   # rotate CW
                    else:
                        tello.send_rc_control(0, FORWARD_SPEED, 0, -rotation_speed)  # rotate CCW
                else:
                    should_move_forward = True
                    tello.send_rc_control(0, FORWARD_SPEED, 0, 0)  # Move forward with constant speed

            max_val_left, max_val_right = check_for_obstacles(depth_map_resized)

            # Threshold for detecting very close objects
            # Set above uint8 range - effectively disables side avoidance.
            # Empirically this worked better: see docstring for explanation.
            close_threshold = CLOSE_THRESHOLD

            # Check for very close objects and react accordingly
            if max_val_left > close_threshold and max_val_right > close_threshold:
                tello.send_rc_control(0, 0, 0, 0)
                print("Close objects detected on both sides. Staying put.")
                should_move_forward = False
            elif max_val_left > close_threshold:
                tello.send_rc_control(0, 0, 0, 40)
                print("Close object detected on the left, rotating right.")
                time.sleep(1)
                should_move_forward = False
            elif max_val_right > close_threshold:
                tello.send_rc_control(0, 0, 0, -40)
                print("Close object detected on the right, rotating left.")
                time.sleep(1)
                should_move_forward = False

            # If no close obstacles detected, move forward
            if should_move_forward:
                tello.send_rc_control(0, FORWARD_SPEED, 0, 0)

            # Display depth map using OpenCV
            cv2.imshow('Depth Map', depth_map_resized)

            # Exit loop if 'q' is pressed
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
