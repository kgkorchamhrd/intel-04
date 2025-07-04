import cv2
import numpy as np
import tensorflow as tf

# 학습한 모델 로드
model = tf.keras.models.load_model("transfer_learning_flower.keras")

# label 이름도 로드
label_name = [
    'dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'
]  # tf_flowers 기준

img_height = 255
img_width = 255

# 웹캠
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 수신 실패")
        break

    # OpenCV는 BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 크기 조절
    resized = cv2.resize(rgb_frame, (img_height, img_width))
    # 전처리
    x = tf.keras.applications.mobilenet_v3.preprocess_input(resized)
    x = np.expand_dims(x, axis=0)  # 배치 차원

    # 예측
    pred = model.predict(x)
    class_idx = np.argmax(pred)
    prob = np.max(pred)

    # 화면에 표시
    cv2.putText(frame, f"{label_name[class_idx]} ({prob:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("flower class", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()