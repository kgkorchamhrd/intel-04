import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.image import resize

# 모델 및 라벨 불러오기
model = load_model('transfer_learning_flower.keras')
(_, _, _), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)
label_names = metadata.features['label'].names
img_height, img_width = 255, 255

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 0번 카메라는 기본 웹캠

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

print("🎥 웹캠이 켜졌습니다. 'q' 키를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임을 읽을 수 없습니다.")
        break

    # 중앙 정사각형 크롭
    h, w, _ = frame.shape
    min_dim = min(h, w)
    cropped = frame[(h - min_dim)//2 : (h + min_dim)//2, (w - min_dim)//2 : (w + min_dim)//2]

    # 전처리
    resized = cv2.resize(cropped, (img_width, img_height))
    img_array = np.expand_dims(resized.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    # 예측
    predictions = model.predict(img_array, verbose=0)
    pred_label = label_names[np.argmax(predictions)]

    # 결과를 화면에 출력
    cv2.putText(frame, f"Prediction: {pred_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Flower Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
