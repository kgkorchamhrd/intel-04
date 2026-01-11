import numpy as np
import cv2

# 실습 1
a = ("A", "B")
b = ("C",)

print(a+b)

# 실습 2
n = int(input("정수 입력 : "))

row = []
data = []

for i in range(1, n*n + 1):
    row.append(i)
    print(i, end=' ')

    if (i % n == 0):
        print()
        data.append(row)
        row = []

total = np.array(data)

print(total)


# 실습 3
print(total.reshape(1 * n*n))

# 실습 4
img = cv2.imread('iu.jpg')
print(img.shape)
expand_img = np.expand_dims(img, axis=0)
print(expand_img.shape)

print(expand_img.transpose((0, 3, 1, 2)).shape)