import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def segment_image(image, k=5):
    image_np = np.array(image)
    (h, w) = image_np.shape[:2]
    pixels = image_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_pixels.reshape((h, w, 3)).astype(np.uint8)

    return segmented_img, pixels, kmeans.labels_, kmeans.cluster_centers_

def plot_3d_color_space(pixels, title, labels=None, centers=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    if labels is None:
        # Plot all pixels as points
        sample_indices = np.random.choice(len(pixels), min(5000, len(pixels)), replace=False)
        sampled_pixels = pixels[sample_indices]
        ax.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], c=sampled_pixels/255, s=5)
    else:
        # Plot pixels colored by cluster labels
        sample_indices = np.random.choice(len(pixels), min(5000, len(pixels)), replace=False)
        sampled_pixels = pixels[sample_indices]
        sampled_labels = labels[sample_indices]
        colors = centers[sampled_labels] / 255
        ax.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], c=colors, s=5)

        # Also plot cluster centers bigger and bold
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=centers/255, s=100, marker='X', edgecolor='k')

    plt.tight_layout()
    return fig

st.set_page_config(page_title="Image Segmentation with K-Means", layout="wide")

st.title("🌈 Interactive Image Segmentation with K-Means Clustering")

uploaded_file = st.file_uploader("Upload an Image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
        """
        <style>
        .block-container {
            padding: 2rem 4rem 2rem 4rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    k = st.slider("Number of segments (clusters)", 2, 10, 5)

    # Segment the image and get pixel info for plots
    segmented_img, pixels, labels, centers = segment_image(image, k)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Segmented Image (k={k})")
        st.image(segmented_img, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        fig_before = plot_3d_color_space(pixels, "3D Color Space (Original Pixels)")
        st.pyplot(fig_before)

    with col4:
        fig_after = plot_3d_color_space(pixels, "3D Color Space (After K-Means)", labels=labels, centers=centers)
        st.pyplot(fig_after)
else:
    st.info("Please upload an image to start segmentation.")
