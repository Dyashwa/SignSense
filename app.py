import streamlit as st
import cv2
import mediapipe as mp
import joblib

st.title("🤟 Sign Language Detector")

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    letter = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y])

            prediction = model.predict([features])
            letter = prediction[0]

    cv2.putText(frame, f"{letter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    FRAME_WINDOW.image(frame)

cap.release()