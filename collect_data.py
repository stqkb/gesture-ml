import cv2
import numpy as np
import mediapipe as mp
import os

SAVE_DIR = "data/raw"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

current_digit = 0
for d in range(10):
    os.makedirs(os.path.join(SAVE_DIR, str(d)), exist_ok=True)

samples = {}
for d in range(10):
    digit_dir = os.path.join(SAVE_DIR, str(d))
    samples[d] = len([f for f in os.listdir(digit_dir) if f.endswith('.npy')])

print("=" * 50)
print("手势数据采集工具")
print("=" * 50)
print("按 0-9 切换数字 | 空格保存 | q 退出")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = None
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_lm.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks, dtype=np.float32)

    info = f"Digit: {current_digit} | Samples: {samples.get(current_digit, 0)}"
    cv2.putText(frame, info, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if landmarks is not None:
        cv2.putText(frame, "Hand OK - Press SPACE", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    else:
        cv2.putText(frame, "No hand", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Collect Gesture Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' ') and landmarks is not None:
        digit_dir = os.path.join(SAVE_DIR, str(current_digit))
        count = len([f for f in os.listdir(digit_dir) if f.endswith('.npy')])
        np.save(os.path.join(digit_dir, f"{count:04d}.npy"), landmarks)
        samples[current_digit] = count + 1
        print(f"  Saved: digit={current_digit}, sample #{count}")
    elif key in range(ord('0'), ord('9') + 1):
        current_digit = key - ord('0')
        print(f"  Switch to digit: {current_digit}")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\nSamples per digit:")
for d in range(10):
    print(f"  {d}: {samples.get(d, 0)}")
