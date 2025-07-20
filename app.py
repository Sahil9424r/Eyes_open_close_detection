from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import threading
from playsound import playsound

app = Flask(__name__)

# Load model and labels
model = load_model("Eyes.keras")
labels = ['Closed', 'Open']

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize camera
camera = cv2.VideoCapture(0)

# Alarm control
alarm_playing = False
eye_closed_start_time = None

def preprocess_eye(eye_img):
    eye_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_resized = cv2.resize(eye_rgb, (256, 256))
    eye_normalized = eye_resized.astype("float32") / 255.0
    return np.expand_dims(eye_normalized, axis=0)

def play_alarm():
    global alarm_playing
    alarm_playing = True
    playsound("alarm.mp3")
    alarm_playing = False

def gen_frames():
    global eye_closed_start_time, alarm_playing

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            eyes_closed = False

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))

                for (ex, ey, ew, eh) in eyes[:1]:  # only predict on first detected eye
                    eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                    if eye_roi.size == 0:
                        continue
                    eye_input = preprocess_eye(eye_roi)
                    pred = model.predict(eye_input, verbose=0)[0][0]
                    label = labels[1] if pred > 0.5 else labels[0]
                    confidence = f"{float(pred):.2f}"

                    if label == 'Closed':
                        eyes_closed = True

                    cv2.putText(frame, f"{label} ({confidence})", (x + ex, y + ey - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                    break  # only one eye

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Check if eyes are closed for > 4 seconds
            if eyes_closed:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                elif time.time() - eye_closed_start_time > 4:
                    if not alarm_playing:
                        threading.Thread(target=play_alarm).start()
            else:
                eye_closed_start_time = None

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    prediction = None
    show_camera = False

    if request.method == 'POST':
        use_camera = request.form.get('use_camera') == 'on'

        if use_camera:
            show_camera = True
        elif 'image' in request.files:
            image_file = request.files['image']
            if image_file:
                img = Image.open(image_file).convert("RGB")
                img = img.resize((256, 256))
                img_array = np.array(img).astype("float32") / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred = model.predict(img_array, verbose=0)[0][0]
                label = labels[1] if pred > 0.5 else labels[0]
                result = f"Prediction from uploaded image: {label} ({float(pred):.2f})"
                prediction = label

    return render_template('index.html', result=result, prediction=prediction, show_camera=show_camera)

if __name__ == '__main__':
    app.run(debug=True)
