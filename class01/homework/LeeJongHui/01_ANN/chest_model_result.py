import numpy as np
import glob
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2


""" dataset load """

model = load_model("./medical_ann.h5")
test_folder = '/home/paper/workspace/intro/chest_xray/chest_xray/test/'
test_normal_filename = test_folder + 'NORMAL/'
test_pneumonia_filename = test_folder + 'PNEUMONIA/'

FileListNormal = glob.glob(test_normal_filename + "*")
FileListPneumonia = glob.glob(test_pneumonia_filename +"*")

normal_len = len(FileListNormal)
pneumonia_len = len(FileListPneumonia)

select_normal = np.random.randint(normal_len, size = 20)
select_pneumonia = np.random.randint(pneumonia_len, size = 20)

print(select_normal)
print(select_pneumonia)


data_test_set = []
for index, positon in enumerate(select_normal):
    data_test_set.append((0, FileListNormal[positon]))

for index, positon in enumerate(select_pneumonia):
    data_test_set.append((1, FileListPneumonia[positon]))


""" 이미지 전처리 """
# 전처리 함수 (OpenCV)
def preprocess_image_cv(path, target_size=(64, 64)):
    img = cv2.imread(path)  # 컬러로 불러오기 (기본은 BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB로 변환
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = img.reshape(1, target_size[0], target_size[1], 3)
    return img

# 20개만 시각화
plt.figure(figsize=(20, 10))  # 4행 5열 배치
for i in range(20):
    label, path = data_test_set[i]

    # 예측
    input_img = preprocess_image_cv(path)
    pred = model.predict(input_img)
    predicted_label = 1 if pred[0][0] >= 0.5 else 0
    prob = pred[0][0]

    # 이미지 로드 (시각화용 원본)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # matplotlib에 맞게 컬러맵 설정
    plt.subplot(4, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # 텍스트로 정답/예측 결과 표시
    true_str = 'Pneumonia' if label == 1 else 'Normal'
    pred_str = 'Pneumonia' if predicted_label == 1 else 'Normal'
    title = f"Correct answer : {true_str}\npredict : {pred_str} ({prob:.2f})"
    plt.title(title, fontsize=10)

plt.tight_layout()
plt.show()
