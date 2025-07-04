import cv2
import numpy as np

img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
kernel = 1/16*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)