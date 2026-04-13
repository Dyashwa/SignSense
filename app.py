import streamlit as st
import joblib
import numpy as np

st.title("🤟 Sign Language Detector")

st.write("⚠️ Webcam is not supported in cloud deployment.")
st.write("This is a demo version of the model.")

model = joblib.load("asl_model.pkl")

st.subheader("Test Model with Dummy Input")

# create random input (since no camera)
dummy_input = np.random.rand(1, 42)

if st.button("Predict"):
    prediction = model.predict(dummy_input)
    st.success(f"Predicted Letter: {prediction[0]}")