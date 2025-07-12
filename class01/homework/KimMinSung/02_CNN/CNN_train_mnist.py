import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.cm as cm
from keras.models import *

# 데이터셋 로드하기
mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
image_train, image_test = f_image_train/255.0, f_image_test/255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f_label_train)

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i], cmap=cm.gray)
    plt.xlabel(class_names[f_label_train[i]])
plt.show()


# 모델 제작
model = Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10개를 분류해야되기 때문에 10개의 노드와 softmax를 줌



model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(image_train, f_label_train, epochs=10, batch_size=128)
model.summary()
model.save('./02_CNN/fashion_mnist.h5')


# model = tf.keras.models.load_model('./02_CNN/fashion_mnist.h5')
# fashion_mnist = tf.keras.datasets.fashion_mnist
# (f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
# f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# num = 10
# predict = model.predict(f_image_test[:num])
# print(f_label_test[:num])
# print(" * Prediction, ", np.argmax(predict, axis = 1))