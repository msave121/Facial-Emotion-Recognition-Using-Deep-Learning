import numpy as np
from keras.models import load_model
import cv2

# Load model
pretrained_model = load_model('best_model.h5')

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Start webcam
video = cv2.VideoCapture(0)

# Haar cascade
faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Force-create window
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame")
        continue

    # Show raw frame FIRST (ensures window appears)
    cv2.imshow("Frame", frame)

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_norm = face_resized / 255.0
        face_input = np.reshape(face_norm, (1, 48, 48, 1))

        # Prediction
        result = pretrained_model.predict(face_input, verbose=0)
        label = np.argmax(result)
        emotion = emotion_labels[label]

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display updated annotated frame
    cv2.imshow("Frame", frame)

    # Quit on key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

video.release()
cv2.destroyAllWindows()
