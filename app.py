import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

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