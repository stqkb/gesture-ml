"""MediaPipe-based hand landmark extraction."""

import cv2
import numpy as np


def extract_landmarks_from_image(image_path: str) -> np.ndarray:
    """
    Extract 21 hand landmarks (63-dim) from an image file.
    Returns (63,) float32 array or raises if no hand detected.
    """
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                           min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hands.close()

    if not results.multi_hand_landmarks:
        raise ValueError(f"No hand detected in {image_path}")

    landmarks = results.multi_hand_landmarks[0]
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features, dtype=np.float32)


def extract_landmarks_from_camera(callback=None, show=True):
    """
    Real-time landmark extraction from webcam.
    callback(landmarks) is called each frame a hand is detected.
    Press 'q' to quit.
    """
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.7, min_tracking_confidence=0.5
    )

    print("📷 Camera started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    if show:
                        mp_draw.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS
                        )

                    features = []
                    for lm in hand_lm.landmark:
                        features.extend([lm.x, lm.y, lm.z])

                    landmarks = np.array(features, dtype=np.float32)
                    if callback:
                        callback(landmarks)

            if show:
                cv2.imshow("Hand Landmarks", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    # Quick test with camera
    def on_detect(landmarks):
        print(f"Hand detected! Shape: {landmarks.shape}, "
              f"Wrist: ({landmarks[0]:.3f}, {landmarks[1]:.3f}, {landmarks[2]:.3f})")

    extract_landmarks_from_camera(callback=on_detect)
