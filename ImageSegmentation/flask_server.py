from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import os
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
import uuid  # <-- Add this at the top with other imports



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Model
model = keras.models.load_model("model.keras")

# Preprocessing
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Segmentation
def segment_image(image_path):
    input_image = preprocess_image(image_path)
    pred_mask = model.predict(input_image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8).squeeze()
    return pred_mask, input_image[0].squeeze()

# Save image
def save_image(array, path):
    array = (array * 255).astype(np.uint8)
    cv2.imwrite(path, array)

# Overlay
def create_overlay(input_image, pred_mask):
    cmap = plt.get_cmap('jet')
    norm_mask = (pred_mask - np.min(pred_mask)) / (np.max(pred_mask) - np.min(pred_mask) + 1e-8)
    color_mask = cmap(norm_mask)[..., :3]
    input_rgb = np.stack([input_image]*3, axis=-1) if input_image.ndim == 2 else input_image
    overlay = (0.6 * input_rgb + 0.4 * color_mask)
    return np.clip(overlay, 0, 1)

# Segment endpoint
@app.route('/segment', methods=['POST'])
def segment():
    image = request.files['image']
    
    # Generate a unique base name using UUID
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}.png"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    # Segment image
    pred_mask, input_image = segment_image(image_path)
    masked_image = input_image * pred_mask
    overlay = create_overlay(input_image, pred_mask)

    # Prepare filenames
    paths = {
        "original": f"/static/{unique_id}_original.png",
        "mask": f"/static/{unique_id}_mask.png",
        "masked": f"/static/{unique_id}_masked.png",
        "overlay": f"/static/{unique_id}_overlay.png"
    }

    # Save images
    save_image(input_image, os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_original.png"))
    save_image(pred_mask, os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_mask.png"))
    save_image(masked_image, os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_masked.png"))
    save_image(overlay, os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_overlay.png"))

    return jsonify(paths)
# Serve UI
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
