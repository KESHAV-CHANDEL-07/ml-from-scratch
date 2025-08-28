from sklearn.datasets import load_digits
import cv2
import matplotlib.pyplot as plt

# Load sample digit (8x8 grayscale, values 0–16)
digits = load_digits()
image = digits.images[0]  # pick first digit
label = digits.target[0]

# Resize to 28x28
image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

plt.imshow(image_resized, cmap="gray")
plt.title(f"Digit: {label}")
plt.show()

cv2.imwrite("digit.png", image_resized)