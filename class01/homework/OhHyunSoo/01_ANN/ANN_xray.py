# 필요한 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image  # 이미지 로드를 위해 추가

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'chest_xray')

# 데이터셋 폴더 구조 확인
mainDIR = os.listdir(DATASET_DIR)
print(mainDIR)

# 데이터셋 경로 설정
train_folder = os.path.join(DATASET_DIR, 'train/')
val_folder = os.path.join(DATASET_DIR, 'val/')
test_folder = os.path.join(DATASET_DIR, 'test/')

# 훈련 데이터 폴더 확인
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'      # 정상 흉부 X-ray 이미지 폴더
train_p = train_folder + 'PNEUMONIA/'   # 폐렴 흉부 X-ray 이미지 폴더

# 정상 이미지 랜덤 선택
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n + norm_pic

# 폐렴 이미지 랜덤 선택
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]  # rand_norm을 rand_p로 수정
sic_address = train_p + sic_pic
print('pneumonia picture title:', sic_pic)

# 선택된 이미지들 로드
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

# 정상 이미지와 폐렴 이미지 시각화
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# CNN 모델 구축 시작
# 주의: 이 모델은 실제로는 단순한 ANN(Artificial Neural Network)입니다
# 컨볼루션 레이어가 없고 Flatten으로 바로 시작합니다

# 입력 레이어 정의 (64x64 RGB 이미지)
model_in = Input(shape=(64, 64, 3))

# 이미지를 1차원으로 평탄화 (64*64*3 = 12288개의 특성)
model = Flatten()(model_in)

# 완전연결층 1: 128개의 뉴런, ReLU 활성화 함수
model = Dense(activation='relu', units=128)(model)

# 출력층: 1개의 뉴런, 시그모이드 활성화 함수 (이진 분류)
model = Dense(activation='sigmoid', units=1)(model)

# 모델 생성
model_fin = Model(inputs=model_in, outputs=model)

# 모델 컴파일
# - 옵티마이저: Adam
# - 손실함수: 이진 교차 엔트로피 (이진 분류용)
model_fin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # metrics 추가

# 테스트 샘플 수와 배치 크기 설정
num_of_test_samples = 600
batch_size = 32

# 이미지 데이터 전처리 및 증강
# 훈련 데이터용 ImageDataGenerator (데이터 증강 포함)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # 픽셀 값을 0-1 사이로 정규화
    shear_range=0.2,          # 전단 변환 (이미지 기울이기)
    zoom_range=0.2,           # 줌 인/아웃
    horizontal_flip=True      # 수평 뒤집기
)

# 검증/테스트 데이터용 ImageDataGenerator (정규화만)
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터 생성기
training_set = train_datagen.flow_from_directory(
    './01_ANN/chest_xray/train',
    target_size=(64, 64),     # 이미지 크기 조정
    batch_size=32,            # 배치 크기
    class_mode='binary'       # 이진 분류 모드
)

# 검증 데이터 생성기
validation_generator = test_datagen.flow_from_directory(
    './01_ANN/chest_xray/val',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 테스트 데이터 생성기
test_set = test_datagen.flow_from_directory(
    './01_ANN/chest_xray/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 모델 구조 출력
model_fin.summary()

# 모델 훈련
cnn_model = model_fin.fit(
    training_set,
    steps_per_epoch=163,           # 에포크당 스텝 수
    epochs=10,                     # 총 에포크 수
    validation_data=validation_generator,
    validation_steps=624           # 검증 스텝 수
)

# 테스트 데이터에서 모델 평가
test_accu = model_fin.evaluate(test_set, steps=624)

# 훈련된 모델 저장
model_fin.save('medical_ann.h5')
print('The testing accuracy is :', test_accu[1]*100, '%')

# 테스트 데이터에 대한 예측 수행
Y_pred = model_fin.predict(test_set, steps=624)  # steps 매개변수 수정
# y_pred = np.argmax(Y_pred, axis=1)  # 이진 분류에서는 argmax 불필요, 주석 처리


# 훈련 과정 시각화 - 정확도 그래프
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')  # 오타 수정
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.pause(2)
plt.clf()

# 훈련 과정 시각화 - 손실 그래프
plt.plot(cnn_model.history['loss'])        # 순서 변경 (훈련 먼저)
plt.plot(cnn_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')  # 레이블 수정
plt.savefig('train_loss.png')
plt.show(block=False)
plt.pause(2)
plt.clf()