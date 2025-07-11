import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import cv2

# GPU 관련 오류 방지를 위해 CPU 사용 강제 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 사전에 훈련된 MobileNetV3 기반 전이학습 모델을 로드
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'transfer_learning_fl.keras'))
image_path = glob.glob(os.path.join(BASE_DIR, 'images', '*.jpg')) + \
             glob.glob(os.path.join(BASE_DIR, 'images', '*.jpeg')) + \
             glob.glob(os.path.join(BASE_DIR, 'images', '*.png'))

label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
print(f"클래스 개수: {len(label_names)}")
print(f"클래스 이름: {label_names}")

img_height, img_width = 255, 255
num = 10
predict_images = []
processed_images = []

for i in range(num):
    image = cv2.imread(image_path[i])                              # BGR 형식으로 읽힘
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                 # RGB 변환
    image_resized = cv2.resize(image, (img_width, img_height))     # 크기 조정
    image_float32 = image_resized.astype(np.float32)               # float32 변환
    image_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(image_float32)
    
    predict_images.append(image_resized)                           # 시각화용 원본 이미지
    processed_images.append(image_preprocessed)                    # 예측용 전처리된 이미지

# 배치로 변환하여 예측
input_batch = np.array(processed_images)
predict = model.predict(input_batch)
predicted_classes = np.argmax(predict, axis=1)

# 시각화
true_labels = []
for i in range(num):
    filename = os.path.basename(image_path[i]).lower()
    true_label = 0  # 기본값
    for j, label in enumerate(label_names):
        if label in filename:
            true_label = j
            break
    true_labels.append(true_label)

# 시각화
plt.figure(figsize=(12, 6))
for i in range(num):
    plt.subplot(2, 5, i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(predict_images[i].astype('uint8'))  # RGB 이미지
    
    true_label = label_names[true_labels[i]]
    pred_label = label_names[predicted_classes[i]]
    color = 'blue' if true_labels[i] == predicted_classes[i] else 'red'
    
    plt.xlabel(f'True: {true_label}\nPred: {pred_label}', color=color)

plt.tight_layout()
plt.show()

# 출력
print("* Prediction:", predicted_classes)
