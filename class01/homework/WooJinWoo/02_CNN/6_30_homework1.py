# 필터 적용하기 실습
import cv2
import numpy as np

img = cv2.imread('iu2.jpg', cv2.IMREAD_GRAYSCALE)
#kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)