import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import copy
import glob

model = keras.models.load_model('./medical_ann.h5')

test_folder = './chest_xray/test/'
test_normal_filename = test_folder+'NORMAL/'
test_pneumonia_filename = test_folder+'PNEUMONIA/'
FileListNormal = glob.glob(test_normal_filename+"*")
FileListPneumonia = glob.glob(test_pneumonia_filename+"*")

normal_len = len(FileListNormal)
pneumonia_len = len(FileListPneumonia)

select_normal = np.random.randint(normal_len, size=20)
select_pneumonia = np.random.randint(pneumonia_len, size=20)

print(select_normal)
print(select_pneumonia)

data_test_sets = []
for index, position in enumerate(select_normal):
    data_test_sets.append((0, FileListNormal[position]))
for index, position in enumerate(select_pneumonia):
    data_test_sets.append((1, FileListPneumonia[position]))


img_size = (64, 64)  # 모델 학습 시 사용한 이미지 크기로 맞춰야 함
y_true = []
y_pred = []
img_list = []

for label, path in data_test_sets:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    img_input = img_resized / 255.0  # 정규화
    img_input = np.expand_dims(img_input, axis=0)  # 배치 차원 추가

    prediction = model.predict(img_input)
    predicted_label = int(prediction[0][0] > 0.5)  # 예: sigmoid 출력 기준 0.5

    y_true.append(label)
    y_pred.append(predicted_label)
    img_list.append(img)

# 이미지 25개 시각화 (5행 5열)
plt.figure(figsize=(20, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(img_list[i])
    plt.title(f"label Target: {y_true[i]} \n  lable Predict: {y_pred[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
