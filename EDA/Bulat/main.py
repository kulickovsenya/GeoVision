import cv2
from scipy.ndimage import rotate as rotate_image

import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('data_source/first_test.png')
if img is None:
    raise ValueError("Image not found or path is incorrect")

image = rotate_image(img, 90)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detector
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use Hough Line Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Create a copy of the original image to draw lines on
line_image = image.copy()

# Draw the vertical lines
if lines is not None:
    for rho, theta in lines[:, 0]:
        # Filter for vertical lines based on the angle theta
        if np.pi / 2 - 0.1 < theta < np.pi / 2 + 0.1:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the original and the line-detected image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Detected Vertical Lines')
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.show()
