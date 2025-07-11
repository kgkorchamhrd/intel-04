import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# 클래스 라벨 (학습할 때 썼던 label_name 순서와 동일해야 함)
label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']  # 예시

# 모델 불러오기
model = tf.keras.models.load_model('transfer_learning_flower.keras')

# 이미지 로드 함수
def predict_image(image_path):
    img = cv2.imread(image_path)  # BGR로 읽힘
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    # 크기 조정 및 전처리
    img_resized = cv2.resize(img, (255, 255))
    img_array = np.expand_dims(img_resized.astype(np.float32), axis=0)
    img_preprocessed = preprocess_input(img_array)

    # 예측
    preds = model.predict(img_preprocessed)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)

    # 결과 출력
    label = f"{label_names[class_idx]} ({confidence * 100:.2f}%)"
    print(f"예측 결과: {label}")

    # 이미지 표시
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow('Image Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 사용 예시
predict_image('./dai.jpg')  # ← 이미지 경로로 수정하세요

