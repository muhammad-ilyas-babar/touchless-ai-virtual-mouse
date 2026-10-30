"""
Muhammad Ilyas Babar Attari Advanced Visual AI Mouse (Master-Level / Pro Engineer Edition)
=====================================================================
Gesture Architecture:
1. ✊ FIST (Mutthi)           : SLEEP / PAUSE Mode (Keyboard typing / relax).
2. ☝️ INDEX FINGER UP         : MOUSE MOVE (1€ Filter - Zero Lag & High Precision).
3. 🤏 THUMB + INDEX PINCH     :
      - 1 Quick Pinch (<0.28s): LEFT CLICK (Fires at 70%).
      - 2 Quick Pinches       : DOUBLE CLICK (Opens apps/folders!).
      - Hold (>0.28s)         : DRAG & DROP (Window/text selection).
4. 👌 THUMB + MIDDLE PINCH    : RIGHT CLICK (Context menu).
5. ✌️ INDEX + MIDDLE UP       : INERTIAL SCROLL (Smooth trackpad scroll).
6. 🤙 CALL SIGN (Thumb+Pinky) : SYSTEM VOLUME (Up = Vol+, Down = Vol-).
"""

import cv2
import numpy as np
import time
import math
import ctypes
import os
import threading
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------
# 1. WINDOWS LOW-LEVEL NATIVE INPUT API
# -------------------------------------------------------------
user32 = ctypes.windll.user32
try:
    user32.SetProcessDPIAware()
except Exception:
    pass

wScr = user32.GetSystemMetrics(0)
hScr = user32.GetSystemMetrics(1)

# Mouse Events
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_WHEEL      = 0x0800

# Keyboard Multimedia Events (Volume)
VK_VOLUME_UP   = 0xAF
VK_VOLUME_DOWN = 0xAE

def set_mouse(x, y):
    cx = max(0, min(wScr - 1, int(x)))
    cy = max(0, min(hScr - 1, int(y)))
    user32.SetCursorPos(cx, cy)

def left_down():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def left_up():
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def right_click():
    user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def scroll_wheel(delta):
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(delta), 0)

def change_volume(up=True):
    vk = VK_VOLUME_UP if up else VK_VOLUME_DOWN
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 2, 0)

# -------------------------------------------------------------
# 2. ZERO-LATENCY THREADED CAMERA PIPELINE
# -------------------------------------------------------------
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

# -------------------------------------------------------------
# 3. PARIS 1€ (ONE EURO) FILTER
# -------------------------------------------------------------
class LowPassFilter:
    def __init__(self):
        self.y = None

    def filter(self, val, alpha):
        if self.y is None:
            self.y = val
            return val
        self.y = alpha * val + (1.0 - alpha) * self.y
        return self.y

class OneEuroFilter2D:
    def __init__(self, min_cutoff=1.1, beta=0.008, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.y_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.dy_filter = LowPassFilter()
        self.last_time = None
        self.prev_x = None
        self.prev_y = None

    def _alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.016
        return 1.0 / (1.0 + tau / te)

    def update(self, x, y, timestamp):
        if self.last_time is None:
            self.last_time = timestamp
            self.prev_x, self.prev_y = x, y
            return x, y

        dt = max(timestamp - self.last_time, 0.001)
        rate = 1.0 / dt
        self.last_time = timestamp

        dx = (x - self.prev_x) / dt
        dy = (y - self.prev_y) / dt
        self.prev_x, self.prev_y = x, y

        a_d = self._alpha(rate, self.d_cutoff)
        edx = self.dx_filter.filter(dx, a_d)
        edy = self.dy_filter.filter(dy, a_d)

        speed = math.hypot(edx, edy)
        cutoff = self.min_cutoff + self.beta * speed

        a = self._alpha(rate, cutoff)
        return self.x_filter.filter(x, a), self.y_filter.filter(y, a)

# -------------------------------------------------------------
# 4. MODEL AUTO-LOADER
# -------------------------------------------------------------
MODEL_FILE = "hand_landmarker.task"
def get_model():
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE
    alt = r"D:\New folder (3)\hand_landmarker.task"
    if os.path.exists(alt):
        return alt
    print("AI Model download ho raha hai...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_FILE)
    return MODEL_FILE

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
)

def overdrive_map(val, min_in, max_in, max_out):
    norm = np.clip((val - min_in) / (max_in - min_in), 0.0, 1.0)
    curve = 0.5 - 0.5 * math.cos(norm * math.pi)
    return curve * max_out

# -------------------------------------------------------------
# 5. MAIN ARCHITECTURAL ENGINE
# -------------------------------------------------------------
def main():
    wCam, hCam = 640, 480
    frameR = 75
    euro_filter = OneEuroFilter2D()

    cam = ThreadedCamera(src=0, width=wCam, height=hCam)

    base_options = python.BaseOptions(model_asset_path=get_model())
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.60,
        min_tracking_confidence=0.55
    )
    detector = vision.HandLandmarker.create_from_options(options)

    last_timestamp = 0
    pTime = time.time()
    bar_percent = 0.0
    bar_y = 400

    # Gesture States
    pinch_start_time = None
    last_click_time = 0.0
    is_dragging = False
    has_clicked = False
    right_click_debounce = False
    prev_scroll_y = None
    prev_vol_y = None

    print("\n" + "="*65)
    print(" 🌟 ADVANCED SCI-FI AI VIRTUAL MOUSE (GOD EDITION)")
    print(" ✊ FIST           : Sleep / Pause Mode")
    print(" ☝️ INDEX          : Fast Mouse Movement")
    print(" 🤏 1x PINCH       : Left Click")
    print(" 🤏🤏 2x PINCH     : Double Click (Open files/apps)")
    print(" ✊ PINCH HOLD     : Drag & Drop")
    print(" 👌 THUMB+MIDDLE   : Right Click")
    print(" ✌️ TWO FINGERS    : Smooth Scrolling")
    print(" 🤙 CALL GESTURE   : Volume Control Up/Down")
    print("="*65 + "\n")

    while True:
        success, img = cam.read()
        if not success or img is None:
            time.sleep(0.005)
            continue

        img = cv2.flip(img, 1)
        curr_time = time.time()
        now_ms = max(int(curr_time * 1000), last_timestamp + 1)
        last_timestamp = now_ms

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect_for_video(mp_image, now_ms)

        # Active Boundary Box
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        status_badge = "STANDBY"
        badge_color = (200, 200, 200)

        if results.hand_landmarks:
            marks = results.hand_landmarks[0]
            lmList = [(int(p.x * wCam), int(p.y * hCam)) for p in marks]

            # Skeleton
            for connection in HAND_CONNECTIONS:
                cv2.line(img, lmList[connection[0]], lmList[connection[1]], (0, 255, 0), 2)
            for pt in lmList:
                cv2.circle(img, pt, 4, (0, 0, 255), cv2.FILLED)

            # Landmarks
            x_thumb, y_thumb = lmList[4]
            x_index, y_index = lmList[8]
            x_mid, y_mid = lmList[12]
            x_pinky, y_pinky = lmList[20]
            wrist = lmList[0]

            palm_size = max(math.hypot(lmList[5][0] - lmList[17][0], lmList[5][1] - lmList[17][1]), 30.0)

            def is_up(tip, pip):
                return math.hypot(lmList[tip][0] - wrist[0], lmList[tip][1] - wrist[1]) > \
                       math.hypot(lmList[pip][0] - wrist[0], lmList[pip][1] - wrist[1]) * 1.15

            index_up = is_up(8, 6)
            middle_up = is_up(12, 10)
            ring_up = is_up(16, 14)
            pinky_up = is_up(20, 18)

            # Left Click Pinch
            pinch_dist = math.hypot(x_index - x_thumb, y_index - y_thumb)
            bar_percent = float(np.clip(np.interp(pinch_dist / palm_size, [0.26, 1.05], [100.0, 0.0]), 0.0, 100.0))
            bar_y = np.interp(bar_percent, [0, 100], [400, 150])

            # Right Click Pinch
            right_pinch = math.hypot(x_mid - x_thumb, y_mid - y_thumb) / palm_size

            # -------------------------------------------------------------
            # FEATURE 1: ✊ MUTTHI (FIST) -> SLEEP / PAUSE MODE
            # -------------------------------------------------------------
            if not index_up and not middle_up and not ring_up and not pinky_up:
                status_badge = "Zzz SLEEP / PAUSED"
                badge_color = (0, 120, 255)
                if is_dragging:
                    left_up()
                    is_dragging = False
                prev_scroll_y = None
                prev_vol_y = None

            # -------------------------------------------------------------
            # FEATURE 2: 🤙 CALL SIGN (Thumb + Pinky) -> VOLUME CONTROLLER
            # -------------------------------------------------------------
            elif not index_up and not middle_up and not ring_up and pinky_up:
                status_badge = "VOLUME CONTROL"
                badge_color = (255, 140, 0)
                cv2.line(img, (x_thumb, y_thumb), (x_pinky, y_pinky), (255, 140, 0), 3)

                curr_vol_y = (y_thumb + y_pinky) // 2
                if prev_vol_y is not None:
                    dy = curr_vol_y - prev_vol_y
                    if abs(dy) > 5:
                        change_volume(up=(dy < 0))
                prev_vol_y = curr_vol_y
                prev_scroll_y = None

            # -------------------------------------------------------------
            # FEATURE 3: 👌 THUMB + MIDDLE -> RIGHT CLICK
            # -------------------------------------------------------------
            elif right_pinch < 0.35 and not index_up:
                status_badge = "RIGHT CLICK"
                badge_color = (0, 165, 255)
                cv2.circle(img, (x_mid, y_mid), 16, (0, 165, 255), cv2.FILLED)
                if not right_click_debounce:
                    right_click()
                    right_click_debounce = True
                prev_scroll_y = None
                prev_vol_y = None

            else:
                right_click_debounce = False
                prev_vol_y = None

                # ---------------------------------------------------------
                # FEATURE 4: ✌️ TWO-FINGER INERTIAL SCROLL
                # ---------------------------------------------------------
                if index_up and middle_up and not ring_up and not pinky_up and bar_percent < 30.0:
                    status_badge = "SMOOTH SCROLL"
                    badge_color = (0, 255, 255)
                    cv2.line(img, (x_index, y_index), (x_mid, y_mid), (0, 255, 255), 3)

                    curr_scroll_y = (y_index + y_mid) // 2
                    if prev_scroll_y is not None:
                        dy = curr_scroll_y - prev_scroll_y
                        if abs(dy) > 2:
                            scroll_wheel(-dy * 16)
                    prev_scroll_y = curr_scroll_y

                # ---------------------------------------------------------
                # FEATURE 5: ☝️ MOVE, SINGLE/DOUBLE CLICK & DRAG
                # ---------------------------------------------------------
                else:
                    prev_scroll_y = None
                    target_x = overdrive_map(x_index, frameR, wCam - frameR, wScr)
                    target_y = overdrive_map(y_index, frameR, hCam - frameR, hScr)
                    filtered_x, filtered_y = euro_filter.update(target_x, target_y, curr_time)

                    # Mouse Freeze during click preparation
                    if bar_percent < 45.0 and not is_dragging:
                        set_mouse(filtered_x, filtered_y)
                        status_badge = "ACTIVE TRACKING"
                        badge_color = (0, 255, 0)
                    elif is_dragging:
                        set_mouse(filtered_x, filtered_y)

                    cx, cy = (x_thumb + x_index) // 2, (y_thumb + y_index) // 2
                    color = (255, 0, 255)

                    # Click / Drag Logic at >= 70%
                    if bar_percent >= 70.0:
                        if pinch_start_time is None:
                            pinch_start_time = curr_time

                        elapsed = curr_time - pinch_start_time

                        if elapsed > 0.28:
                            # HOLD DETECTED -> DRAG MODE
                            if not is_dragging:
                                left_down()
                                is_dragging = True
                            status_badge = "DRAG / SELECTION"
                            badge_color = (0, 140, 255)
                            color = (0, 140, 255)
                        else:
                            status_badge = "CHARGED (CLICK)"
                            badge_color = (0, 255, 0)
                            color = (0, 255, 0)

                    else:
                        # Released
                        if pinch_start_time is not None:
                            elapsed = curr_time - pinch_start_time
                            if is_dragging:
                                left_up()
                                is_dragging = False
                            elif elapsed <= 0.28 and not has_clicked:
                                # Double Click Detection (Within 0.35 seconds)
                                if (curr_time - last_click_time) < 0.35:
                                    left_down()
                                    left_up()
                                    status_badge = "DOUBLE CLICK!"
                                    badge_color = (255, 255, 0)
                                    last_click_time = 0.0
                                else:
                                    left_down()
                                    left_up()
                                    last_click_time = curr_time
                                has_clicked = True
                            pinch_start_time = None

                        if bar_percent < 45.0:
                            has_clicked = False

                    # Indicator Circles
                    cv2.line(img, (x_thumb, y_thumb), (x_index, y_index), color, 3)
                    cv2.circle(img, (x_thumb, y_thumb), 12, color, cv2.FILLED)
                    cv2.circle(img, (x_index, y_index), 12, color, cv2.FILLED)
                    cv2.circle(img, (cx, cy), 14, color, cv2.FILLED)

        else:
            prev_scroll_y = None
            prev_vol_y = None
            if is_dragging:
                left_up()
                is_dragging = False
            pinch_start_time = None
            bar_percent = max(0.0, bar_percent - 15.0)
            bar_y = np.interp(bar_percent, [0, 100], [400, 150])
            status_badge = "NO HAND"
            badge_color = (0, 0, 255)

        # -------------------------------------------------------------
        # 6. HUD / GRAPHICAL GAUGES
        # -------------------------------------------------------------
        # Left Power Meter
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        thresh_y = int(np.interp(70, [0, 100], [400, 150]))
        cv2.line(img, (45, thresh_y), (90, thresh_y), (0, 0, 255), 2)
        cv2.putText(img, "70%", (95, thresh_y + 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        fill_color = (0, 140, 255) if is_dragging else ((0, 255, 0) if bar_percent >= 70.0 else (255, 0, 0))
        cv2.rectangle(img, (50, int(bar_y)), (85, 400), fill_color, cv2.FILLED)
        cv2.putText(img, f'{int(bar_percent)}%', (40, 435), cv2.FONT_HERSHEY_PLAIN, 2, fill_color, 2)

        # FPS & Mode Badge
        fps = 1.0 / (curr_time - pTime) if (curr_time - pTime) > 0 else 60.0
        pTime = curr_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 45), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2)
        cv2.putText(img, f"MODE: {status_badge}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1.4, badge_color, 2)

        cv2.imshow("Murtaza AI Mouse (God Edition)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
