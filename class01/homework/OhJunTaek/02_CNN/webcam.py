import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# 클래스 이름 로드 (직접 코드에서 정의하거나, 파일로 저장한 경우 불러오기)
label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']  # 예시: tf_flowers 기준

# 하이퍼파라미터 (학습 시 사용한 크기와 동일해야 함)
img_height = 255
img_width = 255

# 모델 로드
model = tf.keras.models.load_model('transfer_learning_flower.keras')

# 웹캠 열기
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠 프레임을 읽을 수 없습니다.")
        break

    # 이미지 전처리: 크기 조정 → float32 변환 → preprocess_input
    img = cv2.resize(frame, (img_width, img_height))
    img_array = np.expand_dims(img, axis=0).astype(np.float32)
    img_array = preprocess_input(img_array)

    # 예측
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    label = f"{label_names[class_idx]} ({confidence * 100:.2f}%)"

    # 결과 표시
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow('Flower Classifier', frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()

