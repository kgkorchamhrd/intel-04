import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('fashion_mnist.keras')
mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train/255, f_image_test/255

num = 10
predict = model.predict(f_image_test[:num])
print(predict)
print(f_label_test[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))