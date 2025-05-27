from flask import Flask, request, send_file, render_template
import os
import cv2
import numpy as np
from io import BytesIO
from imagescanner import scan_document

app = Flask(__name__)
UPLOAD_FOLDER = 'pictures'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Your upload form HTML

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return "No image file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read image using OpenCV

    # Pass to your scan_document function
    scanned_img = scan_document(filepath)

    # Convert scanned numpy array to PNG bytes
    _, buffer = cv2.imencode('.png', scanned_img)
    io_buf = BytesIO(buffer)

    # Send image back to client
    return send_file(io_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
