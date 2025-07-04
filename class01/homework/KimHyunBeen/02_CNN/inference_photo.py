import cv2
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model("transfer_learning_flower.keras")

# tf_flowers 라벨 (5개)
label_name = [
    'dandelion', 'daisy', 'tulips', 'sunflowers', 'roses'
]

img_height = 255
img_width = 255

# 테스트 이미지
image_path = "flower_test.jpg"

# 이미지 읽기
img = cv2.imread(image_path)    
if img is None:
    print(f"{image_path} 를 찾을 수 없습니다.")
    exit()

# OpenCV BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (img_height, img_width))

# MobileNetV3 전처리
x = tf.keras.applications.mobilenet_v3.preprocess_input(resized)
x = np.expand_dims(x, axis=0)

# 예측
pred = model.predict(x)
class_idx = np.argmax(pred)
prob = np.max(pred)

print(f"예측 결과: {label_name[class_idx]} ({prob:.2f})")

# 이미지에 예측값 표시
cv2.putText(img, f"{label_name[class_idx]} ({prob:.2f})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("flower class", img)
cv2.waitKey(0)
cv2.destroyAllWindows()