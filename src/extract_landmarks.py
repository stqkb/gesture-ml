"""MediaPipe-based hand landmark extraction."""
import logging, cv2, numpy as np
logger = logging.getLogger(__name__)

def extract_landmarks_from_image(image_path):
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    hands.close()
    if not results.multi_hand_landmarks:
        raise ValueError(f"No hand detected in {image_path}")
    features = []
    for lm in results.multi_hand_landmarks[0].landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features, dtype=np.float32)

def extract_landmarks_from_camera(callback=None, show=True):
    import mediapipe as mp
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    logger.info("Camera started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    if show:
                        draw.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
                    feats = []
                    for lm in hl.landmark:
                        feats.extend([lm.x, lm.y, lm.z])
                    if callback:
                        callback(np.array(feats, dtype=np.float32))
            if show:
                cv2.imshow("Hand Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    def on_detect(lm):
        logger.info(f"Hand detected! Shape: {lm.shape}")
    extract_landmarks_from_camera(callback=on_detect)