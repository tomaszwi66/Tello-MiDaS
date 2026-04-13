# Tello-MiDaS - Experimental Indoor Navigator

> A proof-of-concept embodied AI agent built from a Ryze Tello drone and an old laptop. No LiDAR. No pre-mapped paths. Just a depth model and a simple rule.

---

## What Is This?

An experimental project exploring whether a minimal rule-based system - powered only by monocular depth estimation - can navigate a real indoor environment autonomously.

**The core insight**: a single heuristic ("fly toward the furthest open space") handles the vast majority of indoor navigation. Everything else is edge-case mitigation.

Started in November 2023 on a battered VAIO laptop. It flew. It also crashed. A lot. That was the point.

---

## How It Works

```
Tello RGB stream -> MiDaS_small -> Relative depth map -> Navigation logic -> Drone commands
```

### Navigation Logic

| Rule | Description |
|---|---|
| **Fly Forward Rule** | Find the darkest region in the depth map (furthest perceived space), compute its center, fly toward it |
| **Don't Crash Rule** | Split map into left/right halves; if bright (close) regions appear on either side, stop and rotate away |
| **Planar Constraint** | Altitude locked after takeoff - the Tello camera faces slightly downward, which causes ceiling collisions without this fix |

### The Breakthrough: From Frog Leaps to Fluid Flight

The first version analyzed -> rotated -> moved -> stopped in discrete steps. Safe, but jerky.

The key improvement was **linking yaw rotation speed to the visual offset of the target from the frame center**. This enabled continuous correction during forward motion, transforming the drone's behavior from stop-and-go jumps into smooth, fluid flight through the apartment.

---

## Repository Structure

```
Tello-MiDaS/
├── tello_midas_navigator.py   # Main navigator - continuous RC control with proportional yaw
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install djitellopy opencv-python torch torchvision numpy
```

- Python 3.8+
- PyTorch (CPU works; CUDA speeds things up significantly)
- Ryze Tello drone connected via Wi-Fi
- MiDaS weights download automatically via `torch.hub`

---

## Running It

1. Connect your computer to the Tello's Wi-Fi network
2. Run:

```bash
python tello_midas_navigator.py
```

3. Press `q` to quit and land the drone

> ⚠️ **Safety note**: Run in a large, open room. Clear furniture from the flight path. Keep your hand near the laptop to hit `q`. The system will crash into things - that's expected behavior during testing.

---

## Known Failure Modes

These are documented intentionally. They informed what needs to be fixed in the next version.

| Failure | Cause |
|---|---|
| **The Corner Trap** | MiDaS interprets room corners as deep open corridors - the drone flies straight into the wall |
| **Invisible Glass** | White walls and windows lack visual texture, making them transparent to the depth model |
| **Drift / Dead Reckoning** | A companion trajectory mapper (based on flight commands + timing) accumulated drift within minutes, making the map unusable |

---

## What This Proved

- A depth-based "fly to the furthest point" heuristic solves most indoor navigation by itself
- Indoor autonomy is not just a perception problem - it's a **grounding problem**: the agent needs to understand its position within the space, not just what it sees
- Monocular relative depth (MiDaS) is fragile on textureless and reflective surfaces
- Dead-reckoning localization fails fast in real environments

---

## Status

| Version | Hardware | Status |
|---|---|---|
| v1.0 | Old VAIO laptop (CPU only) | ✅ Archived - works, crashes |
| v2.0 | Lenovo Legion Pro 7 / RTX 4080 | 🔄 In progress - replacing MiDaS with metric depth, adding proper visual odometry |

v2.0 will address the corner trap and localization failures using MoGe (metric depth) and SLAM-based pose estimation.

---

## Citation / Background

This project is a personal research experiment, not a production system. If you build on it, a mention is appreciated.

Related work that informed v2.0 planning:
- [MiDaS](https://github.com/intel-isl/MiDaS) - Intel ISL relative depth model
- [MoGe](https://github.com/microsoft/MoGe) - Microsoft metric monocular geometry (planned for v2)
- [djitellopy](https://github.com/damiafuentes/DJITelloPy) - Tello Python SDK

---

## Author

**Tomasz Wietrzykowski**
- Portfolio: [tomaszwi66.github.io](https://tomaszwi66.github.io/)
- LinkedIn: [linkedin.com/in/tomasz-wietrzykowski](https://www.linkedin.com/in/tomasz-wietrzykowski)

---

## License

MIT - use freely, fly responsibly.
