import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

label = input("Enter letter (A-Z): ").upper()
file_name = "sign_data.csv"
file_exists = os.path.isfile(file_name)

with open(file_name, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["label"] + [f"f{i}" for i in range(42)])

    print("Press S to save sample, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                landmarks = []
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x - wrist_x, lm.y - wrist_y])

        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and result.multi_hand_landmarks:
            writer.writerow([label] + landmarks)
            print("Saved sample")
        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
