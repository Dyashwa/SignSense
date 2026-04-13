import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import mediapipe as mp
import numpy as np
import joblib

st.title("🤟 Real-Time Sign Language Detector")

model = joblib.load("asl_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

class SignDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        import cv2
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        letter = "No Hand"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y])

                prediction = model.predict([features])
                letter = prediction[0]

        cv2.putText(img, f"{letter}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

        return img

webrtc_streamer(key="sign-detection", video_transformer_factory=SignDetector)