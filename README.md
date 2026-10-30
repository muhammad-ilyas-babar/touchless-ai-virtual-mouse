# 🖱️ Real-Time Touchless HCI System (AI Virtual Mouse)

> An asynchronous, latency-optimized computer vision system for touchless Windows desktop interaction using MediaPipe and adaptive signal processing.

<!-- YAHAN TUMHARI VIDEO / GIF AAYEGI -->
<p align="center">
  <img src="assets/demo.gif" alt="System Live Demo" width="750">
</p>

---

## ⚡ Key Engineering Features

- **Asynchronous Multithreaded Pipeline:** 60 FPS real-time capture bypassing camera frame buffer delay.
- **1€ (One-Euro) Filter:** Eliminates 90%+ hand micro-tremors without introducing drag latency.
- **Scale-Invariant Kinematics:** Normalizes pinch distance using palm width (MCP 5 to 17) for accurate depth tolerance.
- **Zero-Drift Deadzone Lock:** Cursor anchors firmly at 35% pinch charge to prevent targeting slip during clicks.

---

## 🖐️ Gesture Control Mapping

| Gesture | Action | Mechanism |
| :--- | :--- | :--- |
| **Index Extended** | Cursor Move | 1€ Filtered Ballistic Tracking |
| **Thumb + Index Pinch** | Left Click / Drag | Fires at 70% Charge (Hold >0.35s to Drag) |
| **Thumb + Middle Pinch** | Right Click | Instant Context Menu |
| **Thumb + Ring Pinch** | Double Click | Instant App / Folder Open |
| **Index + Middle Extended** | Smooth Scroll | Two-finger Trackpad Emulation |
| **Closed Fist** | Sleep / Pause | Disables tracking while resting |

---

## 🛠️ Installation & Setup

1. **Repository Clone karein:**
