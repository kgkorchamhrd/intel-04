import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import *

# 데이터셋 로드하기
mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()
image_train, image_test = image_train/255.0, image_test/255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])
plt.show()


# 모델 제작
model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10개를 분류해야되기 때문에 10개의 노드와 softmax를 줌

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(image_train, label_train, epochs=10, batch_size=50)
model.summary()
model.save('mnist.h5')


model = tf.keras.models.load_model('./mnist.h5')
mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()
image_train, image_test = image_train / 255.0, image_test / 255.0
num = 10
predict = model.predict(image_test[:num])
print(label_test[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))