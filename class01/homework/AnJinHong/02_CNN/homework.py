import cv2
import numpy as np

# 1. Load grayscale image
img = cv2.imread('lena.jpeg', cv2.IMREAD_GRAYSCALE)

# 2. Define custom kernels
kernels = {
    'Identity': np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),
    
    'Ridge Detect': np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]]),
    
    'Sharpen': np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]]),
    
    'Box Blur': np.ones((3, 3), np.float32) / 9.0,
}

# 3. Store output images with labels
outputs = []

for name, kernel in kernels.items():
    filtered = cv2.filter2D(img, -1, kernel)
    labeled = cv2.putText(filtered.copy(), name, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    outputs.append(labeled)

# 4. Gaussian Blurs using OpenCV
gaussian_3x3 = cv2.GaussianBlur(img, (3, 3), 0)
gaussian_5x5 = cv2.GaussianBlur(img, (5, 5), 0)

# 5. Add Gaussian blur outputs with labels
for blur_name, blur_img in [('Gaussian 3x3', gaussian_3x3), ('Gaussian 5x5', gaussian_5x5)]:
    labeled = cv2.putText(blur_img.copy(), blur_name, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2, cv2.LINE_AA)
    outputs.append(labeled)

# 6. Arrange images (3 per row)
rows = []
for i in range(0, len(outputs), 3):
    row_imgs = outputs[i:i+3]
    if len(row_imgs) < 3:
        # Pad with black images if needed
        for _ in range(3 - len(row_imgs)):
            row_imgs.append(np.zeros_like(img))
    rows.append(np.hstack(row_imgs))

# 7. Stack rows vertically
final = np.vstack(rows)

# 8. Show result
cv2.imshow('All Filters', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
