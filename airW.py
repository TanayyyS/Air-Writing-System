import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
draw_color = (255, 255, 255)  # white default


def count_fingers(hand_landmarks):
    fingers = []

    # Thumb (works for right hand)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)

            h, w, c = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # 1 Finger → Draw
            if finger_count == 1:
                cv2.circle(frame, (x, y), 8, draw_color, -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y),
                         (x, y), draw_color, 8)

                prev_x, prev_y = x, y

            # 2 Fingers → Change Color
            elif finger_count == 2:
                draw_color = (0, 255, 0)  # Green

            # 5 Fingers → Clear Screen
            elif finger_count == 5:
                canvas = np.zeros_like(frame)

            # Fist or other gestures → Stop Drawing
            else:
                prev_x, prev_y = 0, 0

    # Merge drawing with webcam
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Writing - Gesture Controlled", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()