import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image # for reading images
# Keras Libraries <- CNN
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from sklearn.metrics import classification_report, confusion_matrix # <- define evaluation metrics

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'
# train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
#Normal pic
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)


# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)
#Let's plt these images
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()
# let's build the CNN model

#ANN
ann = Sequential()
#Convolution
model_in = Input(shape = (64, 64, 3))
model = Flatten()(model_in)
# Fully Connected Layers
model = Dense(activation = 'relu', units = 128) (model)
model = Dense(activation = 'sigmoid', units = 1)(model)
# Compile the Neural network
model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])

num_of_test_samples = 600
batch_size = 32
# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as
# your CNN is getting ready to process that image
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) #Image normalization.

training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./chest_xray/val/',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
model_fin.summary()

ann_model = model_fin.fit(training_set,
                          steps_per_epoch = 163,
                          epochs = 10,
                          validation_data = validation_generator,
                          validation_steps = 624)

test_accu = model_fin.evaluate(test_set, steps = 624)
model_fin.save('medical_ann.h5')
print('The testing accuracy is : ',test_accu[1]*100, '%')
Y_pred = model_fin.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

# 1. 하나의 그림(figure)에 두 개의 서브플롯(subplot)을 나란히 생성
plt.figure(figsize=(12, 5))

# 2. 첫 번째 서브플롯: 모델 정확도 (Model Accuracy)
plt.subplot(1, 2, 1)
plt.plot(ann_model.history['accuracy'], label='Training set')
plt.plot(ann_model.history['val_accuracy'], label='Validation set')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# 3. 두 번째 서브플롯: 모델 손실 (Model Loss)
plt.subplot(1, 2, 2)
# 원본 코드의凡例 순서와 내용을 그대로 유지
plt.plot(ann_model.history['val_loss'], label='Training set') 
plt.plot(ann_model.history['loss'], label='Test set')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# 4. 완성된 그래프 출력
plt.tight_layout() # 그래프 간격 자동 조정
plt.show()

# --- 1) NORMAL 클래스만 파일 경로 수집 ---
normal_folder = './chest_xray/test/NORMAL/'
filepaths = [os.path.join(normal_folder, f)
             for f in os.listdir(normal_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2) 랜덤으로 25개 선택 ---
idxs = random.sample(range(len(filepaths)), 25)

# --- 3) 5×5 그리드 생성 및 출력 ---
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
for ax, idx in zip(axes.flatten(), idxs):
    # 이미지 로드 및 전처리
    img = load_img(filepaths[idx], target_size=(64, 64))
    x   = img_to_array(img) / 255.0

    # 모델 예측
    pred_prob  = model_fin.predict(x[np.newaxis, ...])[0][0]
    pred_label = 'PNEUMONIA' if pred_prob > 0.5 else 'NORMAL'

    # 시각화
    ax.imshow(img)
    ax.axis('off')
    # 두 줄 타이틀: 위는 고정 NORMAL, 아래는 예측 결과
    ax.set_title(f"Label target: NORMAL\nLabel predict: {pred_label}",
                 fontsize=9, pad=6)

plt.tight_layout()
plt.show()