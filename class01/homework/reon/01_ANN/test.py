import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


model = tf.keras.models.load_model('./mnist.h5')
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255, X_test / 255

num = 10
num_list = np.random.randint(0, len(X_test), size=num).tolist()

print(num_list)

for i in num_list :
    predict = model.predict(np.expand_dims(X_test[i], axis=0))
    print(" * Prediction, ", np.argmax(predict, axis = 1), y_test[i])
