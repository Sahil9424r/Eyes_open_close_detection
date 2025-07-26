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

- Developed using **Keras** with the **TensorFlow** backend.
- The model is a **deep Convolutional Neural Network (CNN)** constructed using the `Sequential` API.
- **Input shape**: `256x256x3` RGB images.
- **Classification Type**: Binary classification – predicts whether the eyes are `Open` or `Closed`.

### 🔧 Architecture Overview

- **Preprocessing**:
  - `Rescaling` layer to normalize pixel values to the range [0, 1].
  - `Data Augmentation` layer includes:
    - Random horizontal and vertical flipping
    - Random rotation (20%)
    - Random zoom (20%)

- **Convolutional Layers**:
  - `Conv2D(16, (3,3), activation='relu')`
  - `Conv2D(32, 3, activation='relu')`
  - `MaxPooling2D()`
  - `Conv2D(64, 3, activation='relu')`
  - `MaxPooling2D()`
  - `Conv2D(128, 3, activation='relu')`
  - `MaxPooling2D()`
  - `Conv2D(256, 3, activation='relu')`
  - `MaxPooling2D()`

- **Fully Connected Layers**:
  - `Flatten()`
  - `Dense(128, activation='relu')`
  - `Dense(64, activation='relu')`
  - `Dense(1, activation='sigmoid')` – gives probability of eye being open.

- 🧾 Final trained model is saved as: **`Eyes.keras`**

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
---

## 📸 Demo

### 🔹 1. Home Page

![Home Page](Eye_pic/Screenshot%20(385).png)

---

### 🔹 2. Image Uploaded with Prediction

![Image Upload and Result](Eye_pic/Screenshot%20(388).png)

---

### 🔹 3. Real-Time Webcam Prediction

![Real-Time Webcam Prediction](Eye_pic/Screenshot%20(387).png)

---
## 🔮 Future Improvements

- Add eye aspect ratio (EAR) and blink detection.
- Integrate duration-based drowsiness alerts.
- Visualize training metrics (accuracy, loss).
- Deploy app to platforms like Render or Docker.

---
