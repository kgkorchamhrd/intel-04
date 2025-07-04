import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 범위를 0~1로 Normalized
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2023-12-04', '2025-01-22')
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]

x = dfx.values.tolist()  # open, high, low, volue
y = dfy.values.tolist()  # close 데이터

window_size = 10
data_x = []
data_y = []

for i in range(len(y) - window_size):
    _x = x[i : i + window_size]
    _y = y[i + window_size]
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

# === RNN 모델 ===
model_rnn = Sequential()
model_rnn.add(SimpleRNN(units=20, activation='tanh',
                    return_sequences=True,
                    input_shape=(window_size,4)))
model_rnn.add(Dropout(0.1))
model_rnn.add(SimpleRNN(units=20, activation='tanh'))
model_rnn.add(Dropout(0.1))
model_rnn.add(Dense(units=1))
model_rnn.summary()

model_rnn.compile(optimizer='adam', loss='mean_squared_error')

history = model_rnn.fit(train_x, train_y, validation_data=(val_x, val_y),
                    epochs=70, batch_size=30)

# === GRU 모델 ===
model_gru = Sequential()
model_gru.add(GRU(units=20, activation='tanh',
              return_sequences=True,
              input_shape=(window_size,4)))
model_gru.add(Dropout(0.1))
model_gru.add(GRU(units=20, activation='tanh'))
model_gru.add(Dropout(0.1))
model_gru.add(Dense(units=1))
model_gru.summary()

model_gru.compile(optimizer='adam', loss='mean_squared_error')

history = model_gru.fit(train_x, train_y, validation_data=(val_x, val_y),
                    epochs=70, batch_size=30)

# === LSTM 모델 ===
model_lstm = Sequential()
model_lstm.add(LSTM(units=20, activation='tanh',
               return_sequences=True,
               input_shape=(window_size,4)))
model_lstm.add(Dropout(0.1))
model_lstm.add(LSTM(units=20, activation='tanh'))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(units=1))
model_lstm.summary()

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

history = model_lstm.fit(train_x, train_y, validation_data=(val_x, val_y),
                    epochs=70, batch_size=30)

pred_rnn = model_rnn.predict(test_x)
pred_gru = model_gru.predict(test_x)
pred_lstm = model_lstm.predict(test_x)

plt.plot(test_y, label='Actual', color='red')
plt.plot(pred_rnn, label='predicted (rnn)', color='blue')
plt.plot(pred_gru, label='predicted (gru)', color='orange')
plt.plot(pred_lstm, label='predicted (lstm)', color='green')

plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

# MAPE : mean(|pred - targ| / targ)
# R2-Score : R^2 = 1 - SSres/SStot -> 1 완벽한 예측, 0 평균값만큼도 못함, R^2 < 0 평균보다 안 좋음