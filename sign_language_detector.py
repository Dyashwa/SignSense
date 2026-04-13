import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y])

            prediction = model.predict([features])
            letter = prediction[0]

    cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Letter: {letter}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

    cv2.imshow("A–Z Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
