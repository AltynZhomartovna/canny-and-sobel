import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()  # Select an image from your local machine

# Load the image in grayscale
image = cv2.imread('22.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filter in X and Y direction
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute gradient magnitude
sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))  # Normalize to 0-255

# Display results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(1, 3, 2), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(1, 3, 3), plt.imshow(sobel_magnitude, cmap='gray'), plt.title('Sobel Magnitude')
plt.show()
