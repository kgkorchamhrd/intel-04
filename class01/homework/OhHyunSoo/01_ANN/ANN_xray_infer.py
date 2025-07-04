import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import glob
import os
from PIL import Image

# === 경로 설정 ===
model = keras.models.load_model("./01_ANN/models/medical_ann.h5")
test_folder = '/home/hyunsoo/workspace/01_ANN/chest_xray/test/'

test_normal_filename = test_folder + 'NORMAL/'
test_pneumonia_filename = test_folder + 'PNEUMONIA/'

FileListNormal = glob.glob(test_normal_filename + "*")
FileListPneumonia = glob.glob(test_pneumonia_filename + "*")

normal_len = len(FileListNormal)
pneumonia_len = len(FileListPneumonia)

select_normal = np.random.randint(normal_len, size=20)
select_pneumonia = np.random.randint(pneumonia_len, size=20)

print(select_normal)
print(select_normal)

data_test_sets = []
for index, position in enumerate(select_normal):
    data_test_sets.append((0, FileListNormal[position]))
for index, position in enumerate(select_pneumonia):
    data_test_sets.append((1, FileListPneumonia[position]))

# === 이미지 시각화 (5행 8열 = 총 40장) ===
fig, axes = plt.subplots(5, 8, figsize=(12, 8))
fig.suptitle("분류 결과 예시", fontsize=12)

for idx, (label, filepath) in enumerate(data_test_sets):
    img = Image.open(filepath).convert("RGB")
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 예측값 얻기
    pred = model.predict(img_array, verbose=0)[0][0]
    pred_label = 1 if pred > 0.5 else 0

    row, col = divmod(idx, 8)
    ax = axes[row, col]
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    gt_text = 'PNEUMONIA' if label == 1 else 'NORMAL'
    pred_text = 'PNEUMONIA' if pred_label == 1 else 'NORMAL'
    color = 'red' if label != pred_label else 'blue'

    ax.set_title(f"GT: {gt_text}\nPred: {pred_text}\n{pred:.2f}", color=color)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# # === 경로 설정 ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(BASE_DIR, 'chest_xray', 'chest_xray', 'test', 'NORMAL')  # 또는 PNEUMONIA
# model_path = os.path.join(BASE_DIR, 'models', 'medical_ann.h5')


# # === 모델 로드 ===
# model = tf.keras.models.load_model(model_path)

# # === 이미지 파일 목록 수집 ===
# files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
# files = sorted(files)[:30]  # 최대 30장

# # === 이미지, 라벨, 예측 결과 저장용 리스트 ===
# images = []
# labels = []
# preds = []

# # === 이미지 전처리 및 예측 ===
# for file in files:
#     img_path = os.path.join(folder_path, file)
#     img = Image.open(img_path).resize((64, 64))
#     img_np = np.array(img) / 255.0

#     if img_np.ndim == 2:
#         img_np = np.stack((img_np,) * 3, axis=-1)  # 흑백 → RGB로

#     input_tensor = np.expand_dims(img_np, axis=0)
#     pred = model.predict(input_tensor, verbose=0)[0][0]

#     # 저장
#     images.append(img)
#     labels.append('NORMAL')  # 폴더 기준 (이 코드는 NORMAL 폴더 대상)
#     preds.append('PNEUMONIA' if pred > 0.5 else 'NORMAL')

# # === 출력 ===
# fig, axes = plt.subplots(5, 6, figsize=(15, 12))
# fig.suptitle('출력 예시 (Homework #5)', fontsize=16)

# for idx, ax in enumerate(axes.flat):
#     if idx < len(images):
#         ax.imshow(images[idx], cmap='gray')
#         ax.set_title(f'Label: {labels[idx]}, Predict: {preds[idx]}', fontsize=9)
#         ax.axis('off')
#     else:
#         ax.axis('off')

# plt.tight_layout()
# plt.subplots_adjust(top=0.90)
# plt.show()
