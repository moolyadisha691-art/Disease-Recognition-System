import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
st.markdown("""
<style>
.stApp {
    background-color: #eaf4f4;
}
</style>
""", unsafe_allow_html=True)

# Load trained model
model = load_model("disease_model.keras")

# Disease names
categories = ["Acne", "Eczema", "Psoriasis"]

# Image size
img_size = 224

# Page title
st.title("Disease Recognition System")

# Description
st.write("Upload a skin image to predict the disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# If image is uploaded
if uploaded_file is not None:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image")

    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize image
    img = cv2.resize(img, (img_size, img_size))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict button
    if st.button("Predict Disease"):

        prediction = model.predict(img)

        result = np.argmax(prediction)

        st.success(
            "Predicted Disease: " + categories[result]
        )