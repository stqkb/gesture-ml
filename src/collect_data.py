"""Collect real gesture data from webcam."""

import os, logging
import cv2, numpy as np, mediapipe as mp
from src.utils import resolve_path

logger = logging.getLogger(__name__)
DATA_DIR = str(resolve_path("data/raw"))
STABILITY_THRESHOLD = 0.02
prev_landmarks = None


def is_stable(current, previous):
    if previous is None:
        return True
    return np.abs(current - previous).mean() < STABILITY_THRESHOLD


def main():
    global prev_landmarks
    os.makedirs(DATA_DIR, exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(DATA_DIR, str(d)), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return

    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    logger.info("Camera started. Press 0-9 to save, q to quit.")

    counts = {}
    for d in range(10):
        dd = os.path.join(DATA_DIR, str(d))
        counts[d] = len([f for f in os.listdir(dd) if f.endswith(".npy")])

    current_lm = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_lm = None

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
                    feats = []
                    for lm in hl.landmark:
                        feats.extend([lm.x, lm.y, lm.z])
                    current_lm = np.array(feats, dtype=np.float32)

            # UI
            cv2.putText(frame, "Press 0-9 to save, q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            for d in range(10):
                cv2.putText(frame, f"{d}: {counts[d]} samples", (10, 65+d*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if current_lm is not None:
                stable = is_stable(current_lm, prev_landmarks)
                status = "STABLE" if stable else "Moving..."
                color = (0,255,0) if stable else (0,200,255)
            else:
                status, color, stable = "No hand", (0,0,255), False
            cv2.putText(frame, status, (frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if ord('0') <= key <= ord('9'):
                digit = key - ord('0')
                if current_lm is not None:
                    if not is_stable(current_lm, prev_landmarks):
                        logger.warning("Hand moving, wait to stabilize...")
                    else:
                        np.save(os.path.join(DATA_DIR, str(digit), f"{counts[digit]:04d}.npy"), current_lm)
                        counts[digit] += 1
                        prev_landmarks = current_lm.copy()
                        logger.info(f"Saved digit {digit} ({counts[digit]} total)")
                else:
                    logger.warning("No hand detected")

            if current_lm is not None:
                prev_landmarks = current_lm.copy()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    logger.info("Collection complete!")
    for d in range(10):
        logger.info(f"  Digit {d}: {counts[d]} samples")


if __name__ == "__main__":
    main()