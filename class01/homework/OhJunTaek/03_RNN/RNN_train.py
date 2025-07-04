import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout


# 범위를 0~1로 normalized
def MinMaxScaler(data):
    " 최솟값 과 최댓값을 이용하여 0~1값으로 변환"
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0 으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)


df = fdr.DataReader('005380', '2024-05-04', '2025-03-22')
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High','Low','Volume']]

x = dfx.values.tolist() #open,high,log,volume 데이터
y = dfy.values.tolist() #close 데이터

#ex) 1월 1일 ~ 1월 10일 까지의 OHLV 데이터로 1월 11일 종가 (CLOSE)예측
#ex) 1월 2일 ~ 1월 11일 까지의 OHLV 데이터로 1월 12일 종가 (CLOSE)예
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size] # 다음날 종가
    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])

test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])

print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)
model = Sequential()
model.add(SimpleRNN(units=20, activation='tanh',
        return_sequences=True, input_shape=(10,4)))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=20, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(train_x, train_y, validation_data = (val_x, val_y),
        epochs = 70, batch_size=30)
