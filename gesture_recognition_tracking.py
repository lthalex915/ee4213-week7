import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # single hand for interaction demo
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def _is_finger_extended(landmarks, tip_id, pip_id):
    # Compare y (and slight x) to estimate extension relative to the hand orientation
    tip = landmarks[tip_id]
    pip = landmarks[pip_id]
    return (tip.y < pip.y - 0.02)  # tip above PIP in image coords for extended finger

def _is_thumb_extended(landmarks):
    # Use x distance from wrist and thumb base to estimate thumb openness
    wrist = landmarks[0]
    tip = landmarks[4]
    mp_base = landmarks[2]
    return abs(tip.x - wrist.x) > abs(mp_base.x - wrist.x) + 0.03

def recognize_gesture(landmarks):
    """
    Minimal two-class recognizer for interaction:
    - OPEN_PALM (Unmute/Play): most fingers extended
    - FIST (Mute/Pause): no fingers extended
    """
    index_ext = _is_finger_extended(landmarks, 8, 6)
    middle_ext = _is_finger_extended(landmarks, 12, 10)
    ring_ext = _is_finger_extended(landmarks, 16, 14)
    pinky_ext = _is_finger_extended(landmarks, 20, 18)
    thumb_ext = _is_thumb_extended(landmarks)

    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext]) + (1 if thumb_ext else 0)

    if extended_count >= 4:
        return "OPEN_PALM"
    if extended_count == 0:
        return "FIST"
    return "UNKNOWN"

def open_camera_with_fallback():
    """
    Try multiple camera backends and indices for robustness (from reference).
    On macOS, prefer AVFoundation.
    """
    candidates = []
    try:
        AVFOUNDATION = cv2.CAP_AVFOUNDATION
    except AttributeError:
        AVFOUNDATION = None

    indices = [0, 1]
    backends = []
    if AVFOUNDATION is not None:
        backends.append(AVFOUNDATION)
    backends.append(None)
    for attr in ["CAP_ANY", "CAP_QT", "CAP_V4L2"]:
        api = getattr(cv2, attr, None)
        if api is not None and api not in backends:
            backends.append(api)

    for b in backends:
        for i in indices:
            candidates.append((i, b))

    for idx, backend in candidates:
        if backend is None:
            cap_try = cv2.VideoCapture(idx)
            backend_name = "DEFAULT"
        else:
            cap_try = cv2.VideoCapture(idx, backend)
            backend_name = str(backend)
        if cap_try.isOpened():
            ok, _ = cap_try.read()
            if ok:
                print(f"[INFO] Opened camera index {idx} with backend {backend_name}")
                return cap_try
            cap_try.release()
    return None

def print_macos_permission_help():
    print("\n[HINT] If the camera window does not appear on macOS:")
    print("1) System Settings > Privacy & Security > Camera.")
    print("2) Ensure 'python' or 'VSCode' has camera permission.")
    print("3) If missing, run once from Terminal to trigger the prompt.")
    print("4) You may need to grant 'Visual Studio Code' camera access.\n")

parser = argparse.ArgumentParser(description="Hand Gesture Interaction Demo")
parser.add_argument("--video", type=str, default=None, help="Optional path to input video. Default uses webcam.")
args = parser.parse_args()

cap = None
if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {args.video}. Falling back to webcam.")
        cap = open_camera_with_fallback()
else:
    cap = open_camera_with_fallback()

if cap is None or not cap.isOpened():
    print("[ERROR] Could not open any camera. Falling back to hand.mp4 if available.")
    cap = cv2.VideoCapture("hand.mp4")
    if not cap.isOpened():
        print("[ERROR] Could not open camera or hand.mp4. Exiting.")
        print_macos_permission_help()
        exit(1)
    else:
        print("[INFO] Playing fallback video hand.mp4")

print("Starting hand interaction demo...")
print("Press 'q' to quit the window.")
print_macos_permission_help()
print("Make clear gestures:")
print("- OPEN_PALM: Unmute / Play")
print("- FIST: Mute / Pause")

# Stabilization buffer and state machine for action toggling
BUFFER_SIZE = 7
gesture_buffer = deque(maxlen=BUFFER_SIZE)

state = "MUTED"  # initial state for demo
last_action_time = 0.0
debounce_ms = 600

def most_common_stable(gestures, min_hits):
    if not gestures:
        return None
    counts = {}
    for g in gestures:
        counts[g] = counts.get(g, 0) + 1
    mc = max(counts, key=counts.get)
    return mc if counts[mc] >= min_hits else gestures[-1]  # fallback to latest if not stable enough

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Video finished!")
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    current = "NO_HAND"
    if results.multi_hand_landmarks:
        # Use the first detected hand for interaction
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        current = recognize_gesture(hand_landmarks.landmark)

    gesture_buffer.append(current)
    # Determine stabilized gesture
    counts = {}
    for g in gesture_buffer:
        counts[g] = counts.get(g, 0) + 1
    stable = max(counts, key=counts.get) if counts else "NO_HAND"

    # Debounced state machine
    now = time.time() * 1000.0
    if stable in ("OPEN_PALM", "FIST") and (now - last_action_time) > debounce_ms:
        if stable == "OPEN_PALM" and state != "UNMUTED":
            state = "UNMUTED"
            last_action_time = now
            print("[ACTION] Unmute/Play")
        elif stable == "FIST" and state != "MUTED":
            state = "MUTED"
            last_action_time = now
            print("[ACTION] Mute/Pause")

    # Visual overlays
    status_text = f"Status: {state}"
    color = (0, 255, 0) if state == "UNMUTED" else (0, 0, 255)
    cv2.rectangle(image, (10, 10), (330, 70), (0, 0, 0), -1)
    cv2.putText(image, status_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(image, "OPEN_PALM=Unmute  FIST=Mute   (q to quit)", (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Hand Interaction Demo', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended")
