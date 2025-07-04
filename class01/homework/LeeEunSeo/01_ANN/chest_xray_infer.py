# Homework #5

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# model load
model = tf.keras.models.load_model('./medical_ann.h5')

# dataset load
test_folder = '/home/les/workspace/chest-xray-pneumonia/chest_xray/test/'

test_n = test_folder + 'NORMAL/'
test_p = test_folder + 'PNEUMONIA/'

n_list = glob.glob(test_n + '*')
p_list = glob.glob(test_p + '*')

n_len = len(n_list)
p_len = len(p_list)
print(n_len) # 234
print(p_len) # 390

select_normal = np.random.randint(n_len, size=20)
select_pneumonia = np.random.randint(p_len, size=20)

# print(select_normal)
# print(select_pneumonia)

data_test_sets = []
for idx, pos in enumerate(select_normal):
    data_test_sets.append((0, n_list[pos]))

for idx, pos in enumerate(select_pneumonia):
    data_test_sets.append((1, p_list[pos]))

print(data_test_sets)

pred = []
test_imgs = []
test_labels = []

for label, path in data_test_sets:
    img = load_img(path, target_size=(64, 64))
    test_img = img_to_array(img)
    test_img = np.expand_dims(test_img, axis=0)  # 배치 차원 추가
    test_img = test_img / 255.0

    prediction = model.predict(test_img, verbose=0)
    pred_class = 1 if prediction[0][0] > 0.5 else 0

    pred.append(pred_class)
    test_imgs.append(img)
    test_labels.append(label)

print(test_labels)

# 시각화
fig, axes = plt.subplots(8, 5, figsize=(15, 15))

for i in range(40):
    row = i // 5
    col = i % 5
    
    # 이미지 표시
    axes[row, col].imshow(test_imgs[i])
    axes[row, col].axis('off')
    
    # 제목 설정 (실제 라벨 vs 예측)
    target = "NORMAL" if test_labels[i] == 0 else "PNEUMONIA"
    predicted = "NORMAL" if pred[i] == 0 else "PNEUMONIA"
    
    
    axes[row, col].set_title(f'Label target: {target}\nLabel predict: {predicted}', 
                            fontsize=10)
plt.tight_layout() 
plt.show()

# Accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

# Loss
plt.plot(model.history['val_loss'])
plt.plot(model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()