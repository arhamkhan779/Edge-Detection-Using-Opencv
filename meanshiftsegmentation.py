# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

# --------------------
# App Title
# --------------------
st.set_page_config(layout="wide")
st.title("🌈 Mean Shift Image Segmentation")

# --------------------
# Image Upload
# --------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # --------------------
    # Load image
    # --------------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    flat_image = image_rgb.reshape((-1, 3))

    # --------------------
    # Mean Shift Clustering
    # --------------------
    with st.spinner("Segmenting image using Mean Shift..."):
        bandwidth = estimate_bandwidth(flat_image, quantile=0.1, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(flat_image)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        segmented_image = cluster_centers[labels].reshape(image_rgb.shape).astype(np.uint8)

    # --------------------
    # Show Original & Segmented Image
    # --------------------
    st.subheader("Original vs Segmented Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_rgb, caption="Original Image", use_container_width=True)
    with col2:
        st.image(segmented_image, caption="Segmented Image (Mean Shift)", use_container_width=True)

    # --------------------
    # 3D Plot Helper Function
    # --------------------
    def plot_rgb_distribution(data, title):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        r, g, b = data[:, 0], data[:, 1], data[:, 2]
        ax.scatter(r, g, b, c=data / 255.0, s=1)
        ax.set_title(title)
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        ax.view_init(30, 200)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf

    # --------------------
    # 3D Pixel Distribution Plots
    # --------------------
    st.subheader("🎨 RGB Pixel Distribution in 3D Space")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Before Clustering**")
        buf1 = plot_rgb_distribution(flat_image, "Before Clustering")
        st.image(buf1)

    with col4:
        st.markdown("**After Mean Shift Clustering**")
        seg_flat = cluster_centers[labels]
        buf2 = plot_rgb_distribution(seg_flat, "After Clustering")
        st.image(buf2)

else:
    st.info("👆 Please upload an image to begin.")
