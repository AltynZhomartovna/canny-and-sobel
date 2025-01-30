import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload() 

image_path = "22.jpg"  # Change to your uploaded filename
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)  # Low threshold: 50, High threshold: 150

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edge Detection')
plt.show()
