import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


model = tf.keras.models.load_model('fashion_mnist.keras')
mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train / 255, X_test / 255

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num = 10
num_list = np.random.randint(0, len(X_test), size=num).tolist()

print(num_list)

for i in num_list :
    predict = model.predict(np.expand_dims(X_test[i], axis=0))
    idx = int(np.argmax(predict, axis = 1))
    print(" * Prediction :", class_names[idx], ",Row :", class_names[y_test[i]])
