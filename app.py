import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
import os

# ID du fichier Google Drive
FILE_ID = "1-1ckAIrf02miY6uyId8rXPOfhn_p-mhb"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_PATH = "ai_image_detector.h5"

@st.cache_resource
def load_model():
    """Télécharge et charge le modèle TensorFlow si nécessaire"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle... ⏳ (Cela peut prendre quelques secondes)"):
            try:
                response = requests.get(DOWNLOAD_URL, stream=True)
                response.raise_for_status()  # Vérifie si la requête a réussi
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors du téléchargement du modèle : {e}")
                return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Charger le modèle UNE SEULE FOIS
model = load_model()

if model is not None:
    st.success("Modèle chargé avec succès ✅")

    # Fonction de prétraitement de l'image
    def preprocess_image(image):
        img = np.array(image)
        if img.ndim == 2:  # Si l'image est en niveaux de gris (2D)
            img = np.stack((img,)*3, axis=-1)  # Convertir en RGB
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
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Image Chargée", use_column_width=True)

            if st.button("Analyser l'Image"):
                with st.spinner("Analyse en cours..."):
                    label, confidence = predict_image(image)
                    st.write(f"### Résultat : {label}")
                    st.write(f"🧠 Confiance du modèle : **{confidence:.2%}**")
        except Exception as e:
            st.error(f"Erreur lors du chargement de l'image : {e}")
