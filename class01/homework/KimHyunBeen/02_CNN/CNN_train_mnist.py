import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import keras 

mnist = tf.keras.datasets.fashion_mnist

(f_image_train,f_label_train), (f_image_test,f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train/255.0 , f_image_test/255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i], cmap= cm.gray)
    plt.xlabel(class_names[f_label_train[i]])
plt.show()


# howork #3
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape=(28, 28,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(f_image_train, f_label_train, epochs=10, batch_size=100)
model.summary()
model.save('mnist.h5')