import cv2
import numpy as np

Img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
kernel = 1/256 *np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])
print(kernel)
output = cv2.filter2D(Img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)