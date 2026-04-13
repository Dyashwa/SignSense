# 🤟 SignSense – Real-Time ASL Sign Language Detector

A real-time American Sign Language (ASL) detection system using **OpenCV**, **MediaPipe**, and **Machine Learning**.

---

## 🚀 Features

* 📷 Real-time hand tracking using MediaPipe
* 🧠 Machine Learning model (Random Forest)
* 🔤 Detects A–Z hand signs
* 💾 Custom dataset collection
* ⚡ Fast and lightweight

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* NumPy / Pandas

---

## 📂 Project Structure

* `collect_data.py` → Collect hand landmark data
* `train_model.py` → Train ML model
* `sign_language_detector.py` → Real-time detection
* `camera_test.py` → Test webcam

---

## 🧪 How to Run (LOCAL SETUP)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Dyashwa/SignSense.git
cd SignSense
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Test your camera (optional)

```bash
python camera_test.py
```

---

### 4️⃣ Collect data (optional)

```bash
python collect_data.py
```

* Enter letter (A–Z)
* Press **S** to save samples

---

### 5️⃣ Train model

```bash
python train_model.py
```

---

### 6️⃣ Run detection

```bash
python sign_language_detector.py
```

---

## 🎥 Demo

(Add your demo video or GIF here)

---

## ⚠️ Note

This project uses webcam access via OpenCV, so it must be run locally.
Cloud deployment is limited due to OpenCV and MediaPipe compatibility issues.

---

## 📊 Model

* Algorithm: Random Forest
* Input: 42 hand landmark features
* Output: A–Z letters

---

## 🤝 Contributing

Pull requests are welcome!

---

## ⭐ Support

If you like this project, give it a ⭐!
