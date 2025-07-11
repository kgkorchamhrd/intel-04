import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'fashion_mnist.h5'))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'mnist.h5'))

# fashion_mnist = tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist


# (f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

num = 10
predict = model.predict(f_image_test[:num])
print(f_label_test[:num])
print("* Prediction, ", np.argmax(predict, axis = 1))