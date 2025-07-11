import tensorflow as tf
from tensorflow.keras.models import *
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model load: MNIST / Fashion MNIST Dataset
# fashion_mnist = tf.keras.datasets.fashion_mnist
mnist = tf.keras.datasets.mnist

# (f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
# normalized images
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()

# model.save(os.path.join(BASE_DIR, 'models', 'fashion_mnist.h5'))
model.save(os.path.join(BASE_DIR, 'models', 'mnist.h5'))