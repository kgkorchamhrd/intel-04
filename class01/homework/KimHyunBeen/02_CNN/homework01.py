import cv2
import numpy as np

img = cv2.imread('lena1.png', cv2.IMREAD_GRAYSCALE)
kernel = 1/9*np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
output = cv2.filter2D(img, -1, kernel)

# 원하는 크기로 resize
output_resized = cv2.resize(output, (800, 600))  # (너비, 높이)

cv2.imshow('edge', output_resized)
cv2.waitKey(0)
