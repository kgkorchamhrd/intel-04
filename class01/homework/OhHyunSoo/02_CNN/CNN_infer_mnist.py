import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'fashion_mnist.keras'))

fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num = 10
predict = model.predict(f_image_test[:num])
predicted_classes = np.argmax(predict, axis=1)

plt.figure(figsize=(12, 6))
for i in range(num):
    plt.subplot(2, 5, i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(f_image_test[i], cmap='gray')

    true_label = class_names[f_label_test[i]]
    pred_label = class_names[predicted_classes[i]]
    color = 'blue' if f_label_test[i] == predicted_classes[i] else 'red'

    plt.xlabel(f'True: {true_label}\nPred: {pred_label}', color=color)

plt.tight_layout()
plt.show()

print(f_label_test[:num])
print("* Prediction, ", np.argmax(predict, axis = 1))