# kospi_infer.py - 주가 예측 및 시각화

import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 코드 맨 위에 추가

# MinMaxScaler 정의 (훈련 시와 동일한 함수)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 데이터 준비 (훈련 시와 동일한 전처리)
print("데이터 로딩 및 전처리 중...")
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]

x = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i:i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)

# 데이터 분할 (훈련 시와 동일한 비율)
train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
test_x = np.array(data_x[train_size + val_size:len(data_x)])
test_y = np.array(data_y[train_size + val_size:len(data_y)])

print(f'테스트 데이터 크기: {test_x.shape}, {test_y.shape}')

# 모델 로드 및 예측
print("모델 로딩 중...")

# 사용할 모델들의 파일 경로
model_paths = {
    'rnn': 'kospi_RNN.h5',
    'gru': 'kospi_GRU.h5', 
    'lstm': 'kospi_LSTM.h5'
}

loaded_models = {}
predictions = {}

# 각 모델 로드 및 예측
for name, path in model_paths.items():
    try:
        model = load_model(path)
        loaded_models[name] = model
        print(f"{name.upper()} 모델 로드 완료: {path}")
        
        # 예측 수행
        pred = model.predict(test_x)
        predictions[name] = pred
        print(f"{name.upper()} 예측 완료")
        
    except Exception as e:
        print(f"{name.upper()} 모델 로드 실패: {e}")

if not loaded_models:
    print("사용 가능한 모델이 없습니다!")
    import os
    print("현재 디렉토리 파일 목록:")
    for file in os.listdir('.'):
        if file.endswith('.h5'):
            print(f"  - {file}")
    exit()

print(f"\n총 {len(loaded_models)}개 모델이 로드되었습니다.")

# 시각화
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))

# 실제 값 (빨간색)
plt.plot(range(len(test_y)), test_y.flatten(), 
         color='red', linewidth=2, label='Actual')

# 각 모델별 예측값
colors = ['blue', 'orange', 'green']
for i, (name, pred) in enumerate(predictions.items()):
    plt.plot(range(len(pred)), pred.flatten(), 
             color=colors[i % len(colors)], linewidth=2, 
             label=f'predicted ({name})')

plt.title('SEC stock price prediction', fontsize=14)
plt.xlabel('time', fontsize=12)
plt.ylabel('stock price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 성능 평가
# 성능 평가 함수들 (sklearn 없이 직접 구현)
def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

print("\n=== 모델 성능 평가 ===")
for name, pred in predictions.items():
    mse = mean_squared_error(test_y, pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(test_y, pred)
    mape = calculate_mape(test_y, pred)
    
    print(f"\n{name.upper()} 모델:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  정확도: {100 - mape:.2f}%")

print("\n예측 및 시각화 완료!")