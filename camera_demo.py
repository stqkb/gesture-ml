import cv2
import numpy as np
import mediapipe as mp
from src.predict import GesturePredictor

predictor = GesturePredictor("models/best_model.pt")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            features = []
            for lm in hand_lm.landmark:
                features.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(features, dtype=np.float32)
            digit = predictor.predict(landmarks)
            proba = predictor.predict_proba(landmarks)
            confidence = max(proba.values())
            cv2.putText(frame, f"Digit: {digit} ({confidence:.1%})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 25, 0), 3)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
