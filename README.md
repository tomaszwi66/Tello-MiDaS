# Tello-MiDaS - Experimental Indoor Navigator

> A proof-of-concept embodied AI agent built from a Ryze Tello drone and an old laptop.
> No LiDAR. No pre-mapped paths. Just a depth model and one rule.

![Demo](TelloGIF.gif)

---

## The Idea

Can a drone navigate a real indoor environment using nothing but a monocular RGB camera and a depth estimation model?

No sensors. No maps. No pre-programmed paths. Just one heuristic:

> **Find the darkest area on the depth map - the furthest open space - and fly toward it.**

That single rule handles the vast majority of indoor navigation. Everything else is edge-case mitigation.

Started in 2024 on a battered VAIO laptop. It flew. It also crashed. A lot. That was the point.

---

## How It Works

```
Ryze Tello RGB stream -> MiDaS_small -> Relative depth map -> Navigation logic -> Drone commands
```

### Navigation Rules

| Rule | Description |
|---|---|
| **Fly Forward** | Find the darkest region in the depth map (furthest perceived space), compute its center, fly toward it |
| **Don't Crash** | Split the depth map into left/right halves - if bright (close) regions appear on either side, stop and rotate away |
| **Planar Constraint** | Altitude locked after takeoff - the Tello camera faces slightly downward, which causes ceiling collisions without this fix |

---

## Two Scripts - Two Behaviors

This repo contains two versions of the navigator. The core logic is identical - the difference is how the drone moves.

### `tello_midas_stepwise.py` - Stop and Go

The drone moves in discrete steps: analyze, rotate, stop, move, stop, repeat.

Uses Tello's high-level commands:
```python
tello.rotate_clockwise(N)   # drone stops, rotates N degrees, stops
tello.move_forward(N)       # drone stops, moves N cm, stops
```

Flight looks like a frog jumping between positions. Safe and predictable, but jerky.

---

### `tello_midas_continuous.py` - Fluid Flight

The drone blends yaw correction and forward motion simultaneously via raw RC control.

Uses continuous RC signals:
```python
tello.send_rc_control(left_right, forward_back, up_down, yaw)  # all axes at once
```

The key insight: linking yaw rotation speed to the visual offset of the target from the frame center enables continuous heading correction *while* moving forward. One small change - the drone stops making rigid frog leaps and starts flying fluidly through the apartment.

---

## Repository Structure

```
Tello-MiDaS/
├── tello_midas_stepwise.py    # Stop-and-go: discrete rotate -> move -> stop
├── tello_midas_continuous.py  # Fluid flight: simultaneous yaw + forward via RC control
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

```
djitellopy
opencv-python
torch
torchvision
numpy
```

- Python 3.8+
- PyTorch - CPU works, CUDA speeds things up significantly
- Ryze Tello connected via Wi-Fi
- MiDaS weights download automatically via `torch.hub` on first run

---

## Running It

1. Connect your computer to the Tello's Wi-Fi network
2. Run either version:

```bash
# Stop-and-go (safer for first tests)
python tello_midas_stepwise.py

# Fluid flight (smoother, recommended)
python tello_midas_continuous.py
```

3. Press `q` to quit and land the drone

> ⚠️ **Safety note**: Run in a large, open room. Clear furniture from the flight path. Keep your hand near the laptop to hit `q`. The system will crash into things - that is expected behavior during testing.

---

## Known Failure Modes

Documented intentionally. They informed what needs to be fixed in the next version.

| Failure | Cause |
|---|---|
| **The Corner Trap** | MiDaS interprets room corners as deep open corridors - the drone flies straight into the wall |
| **Invisible Glass** | White walls and windows lack visual texture, making them effectively invisible to the depth model |
| **Dead Reckoning Drift** | A companion trajectory mapper based on flight commands and timing accumulated drift within minutes, making the map unusable |

---

## What This Proved

- A depth-based "fly to the furthest point" heuristic solves most indoor navigation by itself
- Indoor autonomy is not just a perception problem - it is a **grounding problem**: the agent needs to understand its position within the space, not just what it sees
- Monocular relative depth (MiDaS) is fragile on textureless and reflective surfaces
- Dead-reckoning localization fails fast in real environments

---

## Status

| Version | Hardware | Status |
|---|---|---|
| v1.0 - Stepwise | Old VAIO laptop, CPU only | Archived - works, crashes |
| v2.0 - Continuous | Old VAIO laptop, CPU only | Archived - smoother flight, same failure modes |
| v3.0 | Lenovo Legion Pro 7 / RTX 4080 | In progress - metric depth, visual odometry, SLAM |

v3.0 will address the corner trap and localization failures using MoGe (metric depth) and SLAM-based pose estimation.

---

## Related Work

- [MiDaS](https://github.com/intel-isl/MiDaS) - Intel ISL relative depth model
- [MoGe](https://github.com/microsoft/MoGe) - Microsoft metric monocular geometry (planned for v3)
- [djitellopy](https://github.com/damiafuentes/DJITelloPy) - Tello Python SDK

---

## Author

**Tomasz Wietrzykowski**
- Portfolio: [tomaszwi66.github.io](https://tomaszwi66.github.io/)
- LinkedIn: [linkedin.com/in/tomasz-wietrzykowski](https://www.linkedin.com/in/tomasz-wietrzykowski)

---

## License

MIT - use freely, fly responsibly.
