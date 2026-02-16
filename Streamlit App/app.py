import streamlit as st
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image

# Define the emotion labels (must match training)
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

@st.cache_resource
def load_emotion_model():
    # Make sure best_model.h5 is in the SAME folder as this file
    model = load_model("best_model.h5")
    return model

pretrained_model = load_emotion_model()

def preprocess_image(uploaded_file, target_size=(48, 48)):
    img = Image.open(uploaded_file)
    img = img.convert("L")              # grayscale
    img = img.resize(target_size)       # 48x48
    img_array = img_to_array(img)
    img_array = img_array / 255.0       # normalize
    reshaped = np.reshape(img_array, (1, 48, 48, 1))
    return reshaped

st.title("Facial Emotion Recognition (Image Upload)")
st.write("Upload a face image and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img_array = preprocess_image(uploaded_file)

    st.write(f"Image tensor shape: {img_array.shape}")

    # Predict
    predictions = pretrained_model.predict(img_array, verbose=0)
    label = int(np.argmax(predictions))
    emotion = emotion_labels[label]

    # Show image + result
    st.image(uploaded_file, caption="Uploaded Image", width=400)
    st.write(f"**Predicted Emotion:** {emotion}")

    # 🔽 Nicely formatted class probabilities
    st.subheader("Class probabilities")
    probs = predictions[0]

    for idx, prob in enumerate(probs):
        emo_name = emotion_labels[idx]
        percentage = prob * 100
        st.write(f"- **{emo_name}**: {percentage:.2f}%")

