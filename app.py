import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
import os

# Installation des dépendances au démarrage
os.system("pip install --upgrade pip")
os.system("pip install streamlit tensorflow numpy opencv-python-headless requests pillow torch torchvision")
# ID du fichier Google Drive
FILE_ID = "1-1ckAIrf02miY6uyId8rXPOfhn_p-mhb"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "ai_image_detector.h5"

@st.cache_resource
def load_model():
    """Télécharge et charge le modèle TensorFlow si nécessaire"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle... ⏳ (Cela peut prendre quelques secondes)"):
            response = requests.get(DOWNLOAD_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

# Charger le modèle UNE SEULE FOIS
model = load_model()

st.success("Modèle chargé avec succès ✅")

# Charger le modèle entraîné
model_path = "ai_image_detector.h5"  # Assure-toi que ton modèle est bien sauvegardé ici
model = tf.keras.models.load_model(model_path)

# Fonction de prétraitement de l'image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Redimensionner l'image
    img = img / 255.0  # Normalisation
    img = np.expand_dims(img, axis=0)  # Ajouter une dimension batch
    return img

# Fonction de prédiction
def predict_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img)[0][0]
    label = "🔴 Image Générée par IA" if prediction > 0.5 else "🟢 Image Authentique"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Interface Streamlit
st.title("Détecteur d'Images Générées par IA 🖼️🤖")
st.write("Charge une image et vérifie si elle est réelle ou générée par une IA.")

# Upload d'image
uploaded_image = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Affichage de l'image
    image = Image.open(uploaded_image)
    st.image(image, caption="Image Chargée", use_column_width=True)

    # Prédiction
    if st.button("Analyser l'Image"):
        label, confidence = predict_image(image)
        st.write(f"### Résultat : {label}")
        st.write(f"🧠 Confiance du modèle : **{confidence:.2%}**")
