import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image


img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE # 병렬연산을 할 것인지에 대한 인자를 처리

# Dataset 준비
(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers',
                                                  split = ['train[:80%]', 'train[80%:90%]', 'train[:90%]'],
                                                  with_info= True,
                                                  as_supervised= True,)

num = 20

def prepare(ds, batch = 1, shuffle = False, augment = False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    #Resize and rescale all datasets
    # x : image, y : label


    # 이미지 크기 조정
    ds = ds.map(lambda x, y : (tf.image.resize(x, [img_height, img_width]), y),
                num_parallel_calls = AUTOTUNE)
    
    # Batch all datasets
    ds = ds.batch(batch_size)

    # 데이터 로딩과 모델 학습이 병렬로 처리되기 위해
    # prefetch()를 사용해서 현재 배치가 처리되는 동안 다음 배치의 데이터를 미르 로드 하도록 함

    return ds.prefetch(buffer_size = AUTOTUNE)



num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes, ", type : ", type(label_name))

test_ds = prepare(test_ds, num)
image_test, label_test = next(iter(test_ds))
image_test = np.array(image_test)
label_test = np.array(label_test, dtype= 'int')

# 모델 불러오기
model = tf.keras.models.load_model('transfer_leraning_flower.keras')
model.summary()

predict = model.predict(image_test)
predicted_classes = np.argmax(predict, axis = 1)


print(" 실제 레이블 | 예측 레이블")
print("----------------------")

for ll in range((label_test.size)):
    print(label_name[label_test[ll]], "|",
          label_name[predicted_classes[ll]])
    
print("----------------------")

# print("실제 레이블:", [label_name[idx] for idx in label_test])
# print("예측 레이블:", [label_name[idx] for idx in predicted_classes])

accuracy = np.mean(predicted_classes == label_test)
print(f"정확도 : {accuracy : .2%}")



# # 모델 및 라벨 로드
# model = tf.keras.models.load_model('transfer_leraning_flower.keras')

# # 클래스 이름 수동 정의 (tf_flowers는 5개 클래스)
# label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# # 이미지 크기
# img_height = 255
# img_width = 255

# # 전처리 함수
# preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

# # ⬇️ 예측할 이미지 경로 설정
# img_path = '/home/paper/workspace/02_CNN/sunflower.jpeg'  # ← 여기에 실제 이미지 경로를 지정

# # 🔄 이미지 로드 및 전처리
# img = Image.open(img_path).convert('RGB')
# img_resized = img.resize((img_width, img_height))
# img_array = np.array(img_resized)

# # 시각화를 위해 복사본 유지
# original_img = img_array.copy()

# # 전처리 및 배치 차원 추가
# img_array = preprocess_input(img_array)
# img_array = np.expand_dims(img_array, axis=0)  # (1, 255, 255, 3)

# # ⏱️ 예측
# predictions = model.predict(img_array)
# predicted_class = np.argmax(predictions[0])
# confidence = np.max(predictions[0])

# # ✅ 출력 및 시각화
# print(f"예측 클래스: {label_names[predicted_class]} (확률: {confidence:.2f})")

# plt.imshow(original_img)
# plt.title(f"what flower?: {label_names[predicted_class]} ({confidence:.2f})")
# plt.axis('off')
# plt.show()
