import numpy as np 
import matplotlib.pyplot as plt 
import os
from PIL import Image 

# Keras Libraries <- CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix # <- define
#evaluation metrics

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_forder = './chest_xray/train/'
val_forder = './chest_xray/val/'
test_folder = './chest_xray/test/'

os.listdir(train_forder)
train_n = train_forder + 'NORMAL/'
train_p = train_forder + 'PNEUMONIA/'

print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic

rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneumonia picture title: ', sic_pic)

norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()


model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (64,64,3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation ='sigmoid'))

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, 
                                   zoom_range =0.2 , horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255) # Image normalization

training_set = train_datagen.flow_from_directory('./chest_xray/train',target_size = (64,64), 
                                                 batch_size = 32, class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory('./chest_xray/val' , target_size = (64,64),
                                                         batch_size =32, class_mode ='binary')
test_set = test_datagen.flow_from_directory('./chest_xray/test', target_size = (64,64),
                                            batch_size =32, class_mode = 'binary')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# homework #4
cnn_model = model.fit(training_set, steps_per_epoch = 163 , epochs =10,validation_data = validation_generator, validation_steps = 624)

test_acuu = model.evaluate(test_set, steps =624)
model.save('./medical_ann.h5')
print('The testing accuracy is : ', test_acuu[1]*100, '%')
Y_pred = model.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

# Accuracy
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validationset'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()
# Loss
plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'],
loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()