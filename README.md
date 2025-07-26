# 👁️ Eyes Open/Close Detection using Deep Learning

This project detects whether a person’s eyes are **Open** or **Closed** in real time using a Convolutional Neural Network (CNN) with OpenCV for face detection and a Flask-based interface for live webcam feed.

---

## 📌 Project Overview

- **Objective:** Classify eye state (open or closed) in real-time webcam input.
- **Tech Stack:** TensorFlow, Keras, OpenCV, Flask.
- **Output:** Displays real-time predictions and triggers alerts when eyes are closed.

---

## 🗃️ Dataset

- **Source:** [MRL Eye State Dataset (Kaggle)](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)
- **Size:** ~4000 images
- **Classes:** `Open`, `Closed`
- Contains eye images from different individuals under varied lighting and conditions.

---

## 🧠 Model Details

- Built using Keras with TensorFlow backend.
- Input shape: 256x256x3 RGB images.
- Binary classifier: `Open`, `Closed`
- Data preprocessing includes normalization and extensive image augmentation (flipping, rotation, zoom).
- Final trained model saved as `Eyes.keras`.

---

## 🧪 Model Training Performance

The model was trained for 10 epochs and showed continuous improvements:

- **Starting Accuracy:** ~52%
- **Final Training Accuracy:** ~95%
- **Final Validation Accuracy:** **96.75%**
- **Final Validation Loss:** 0.0731

This indicates strong generalization and good model reliability on unseen eye images.

---

## 🔍 Prediction Insights

After training, sample predictions from the training set showed correct classification of eye states along with high confidence scores. The model is able to distinguish subtle eye features reliably.

---

## 🖥️ Application Features

- Detects faces using Haar Cascade.
- Automatically crops the eye regions.
- Predicts each eye's state using the trained model.
- Shows prediction label (Open/Closed) in real time.
- Plays a warning sound if eyes are detected as closed.

---

## 📦 Requirements

- Flask  
- OpenCV  
- TensorFlow  
- Numpy  
- Pillow  
- Playsound

---
## ⚙️ Setup & Execution

- Clone the repo.
- Install dependencies in a virtual environment.
- Run the Flask server.
- Open your browser to view the real-time prediction system.

---


## 🔮 Future Improvements

- Add eye aspect ratio (EAR) and blink detection.
- Integrate duration-based drowsiness alerts.
- Visualize training metrics (accuracy, loss).
- Deploy app to platforms like Render or Docker.

---
