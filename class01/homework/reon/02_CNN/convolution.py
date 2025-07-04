import cv2
import numpy as np
from filter import Filter
import argparse

parser = argparse.ArgumentParser(description = "Select a filter")
parser.add_argument('--f', type=str, required=True, help='Filter name to apply')
args = parser.parse_args()

f = Filter()
name = args.f
img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
kernel = f.handle(name)
if isinstance(kernel, str):
    print(f"Error: {kernel}")
    exit(1)
print(kernel)
output = cv2.filter2D(img, -1,kernel)
cv2.imshow(name, output)
cv2.waitKey(0)
