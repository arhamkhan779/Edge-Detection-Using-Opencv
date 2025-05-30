import streamlit as st
st.set_page_config(page_title="Lung X-ray Segmentation", layout="wide")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

# Load model only once
@st.cache_resource
def load_model(model_path="model.keras"):
    return keras.models.load_model(model_path)

model = load_model()

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def segment_image(image_path):
    input_image = preprocess_image(image_path)
    pred_mask = model.predict(input_image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()
    return pred_mask, input_image[0].squeeze()

def mask_overlay_original(image_path):
    pred_mask, input_image = segment_image(image_path)
    multiply_image = input_image * pred_mask

    # Normalize mask and apply jet colormap
    cmap = plt.get_cmap('jet')
    norm_mask = (pred_mask - np.min(pred_mask)) / (np.max(pred_mask) - np.min(pred_mask) + 1e-8)
    color_mask = cmap(norm_mask)[..., :3]

    # Convert input image to RGB if grayscale
    input_rgb = np.stack([input_image]*3, axis=-1)
    overlay = (0.6 * input_rgb + 0.4 * color_mask)
    overlay = np.clip(overlay, 0, 1)

    return input_image, pred_mask, multiply_image, overlay

# ----------------- UI -----------------
st.title("🫁 Lung X-ray Segmentation using U-Net")
st.markdown("Upload a chest X-ray to visualize the **predicted lung segmentation**.")

uploaded_file = st.file_uploader("Upload X-ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # Save uploaded file temporarily
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image.resize((256, 256)))
    image_path = "temp_input_image.png"
    cv2.imwrite(image_path, image_np)

    # Get all output images
    original, pred_mask, multiply_image, overlay = mask_overlay_original(image_path)

    # Create columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(original, caption="Original X-ray", use_container_width=True, clamp=True)

    with col2:
        st.image(pred_mask * 255, caption="Predicted Mask", use_container_width=True, clamp=True)

    with col3:
        st.image(multiply_image, caption="Masked Image", use_container_width=True, clamp=True)

    with col4:
        st.image(overlay, caption="Overlay on Original", use_container_width=True, clamp=True)
