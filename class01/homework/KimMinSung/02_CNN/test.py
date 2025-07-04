import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds

# 1. 모델 로드
model = load_model('transfer_learning_flower.keras')

# 2. 클래스 라벨 이름 가져오기
_, metadata = tfds.load('tf_flowers', split='train', with_info=True, as_supervised=True)
label_names = metadata.features['label'].names

# 3. 전처리 함수
def preprocess_image(frame):
    img = cv2.resize(frame, (255, 255))
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    img = np.expand_dims(img, axis=0)  # (1, 255, 255, 3)
    return img

# 4. 웹캠 시작
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리 및 예측
    img_input = preprocess_image(frame)
    pred = model.predict(img_input)
    class_id = np.argmax(pred)
    confidence = np.max(pred)
    label_text = f"{label_names[class_id]} ({confidence * 100:.1f}%)"

    # 결과 출력
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Flower Detection", frame)

    # 종료 조건 (ESC 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료
cap.release()
cv2.destroyAllWindows()
