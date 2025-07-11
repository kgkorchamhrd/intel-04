import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Conv2D
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pickle

img_height = 255
img_width = 255
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes, ", type : ", type(label_name))

# dataset 준비
def prepare(ds, batch=1, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input

    # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
                num_parallel_calls=AUTOTUNE)
    
    # 전처리 적용
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    # Batch all datasets
    ds = ds.batch(batch_size)

    if augment:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and _vertical"),
            layers.RandomRotation(0.2),
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
        
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)



''' 전이학습 '''
# MobileNet이라는 모델을 사용하겠다.
base_model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet', # imagenet으로 학습된 가중치를 사용
    input_shape=(img_height, img_width, 3),
    
    # 원래 imagenet의 클래스 분류는 1000개인데
    # 우리의 데이터에 맞게 5개의 클래스로 분류되도록 하기 위함
    include_top = False
)
base_model.trainable = False # 내부 파라미터 동결

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=15, validation_data=val_ds)
model.save('transfer_learing_flowers.keras')
with open('history_flower', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

''' 정확도 평가 '''
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy : {test_acc * 100:.2f}%")


''' 예측값 실제 값 시각화 '''
# 클래스 이름
class_names = label_name

# 예측 실행
predictions = model.predict(test_ds)
pred_labels = np.argmax(predictions, axis=1)

# 실제 라벨 추출
true_labels = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

# 이미지 데이터 추출 (시각화용)
images = np.concatenate([x.numpy() for x, y in test_ds], axis=0)

# 일부 예시만 시각화
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].astype("uint8"))
    plt.title(f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[true_labels[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()