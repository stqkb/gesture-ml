"""Real-time gesture recognition with webcam."""
import time, logging
import cv2, numpy as np, mediapipe as mp
from src.predict import GesturePredictor

logger = logging.getLogger(__name__)
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

def main():
    predictor = GesturePredictor()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    logger.info("Camera started, press 'q' to quit")
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
                    feats = []
                    for lm in hl.landmark:
                        feats.extend([lm.x, lm.y, lm.z])
                    lm = np.array(feats, dtype=np.float32)
                    digit = predictor.predict(lm)
                    proba = predictor.predict_proba(lm)
                    cv2.putText(frame, f"Digit: {digit} ({proba[str(digit)]:.1%})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elapsed = time.time() - t0
            if elapsed < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - elapsed)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()