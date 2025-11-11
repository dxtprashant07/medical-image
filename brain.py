import tensorflow as tf
import numpy as np
import cv2
import h5py
import os
import matplotlib.pyplot as plt

# Load Trained Model
MODEL_PATH = "brain_best_model (1).keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Set image size (same as training)
IMAGE_SIZE = (256, 256)

# Function to preprocess a single image (supports both .mat and standard image formats)
def preprocess_image(image_path):
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext == ".mat":
        # Load .mat file
        with h5py.File(image_path, 'r') as data:
            img = np.array(data['cjdata']['image'], dtype=np.float32)
            mask = np.array(data['cjdata']['tumorMask'], dtype=np.uint8)
    else:
        # Load standard image formats (.jpg, .png, .webp, etc.)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        mask = np.zeros_like(img)  # No ground-truth mask for non-.mat images

    # Resize and normalize
    img = cv2.resize(img, IMAGE_SIZE) / 255.0
    mask = cv2.resize(mask, IMAGE_SIZE)
    mask = (mask > 0).astype(np.uint8)  # Ensure binary mask

    # Add channel dimension
    img = img[..., np.newaxis]
    mask = mask[..., np.newaxis]
    
    return img, mask

# Select Image for Testing
test_image_path = "1.mat"  # Change this to your test file

# Load and preprocess image
test_img, test_mask = preprocess_image(test_image_path)

# Predict using the model
test_img_input = np.expand_dims(test_img, axis=0)  # Add batch dimension
pred_mask = model.predict(test_img_input)[0]  # Get first output

# Convert predicted mask to binary
pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

# Plot original image, true mask (if available), and predicted mask
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_img.squeeze(), cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(test_mask.squeeze(), cmap='gray')
plt.title("True Mask (if available)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask_bin.squeeze(), cmap='gray')
plt.title("Predicted Mask")
plt.axis("off")

plt.show()
