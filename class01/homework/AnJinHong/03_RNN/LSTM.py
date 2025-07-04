import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

# MinMaxScaler
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 주식 데이터 로드
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', 'High', 'Low','Volume','Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High','Low','Volume']]
x = dfx.values.tolist()
y = dfy.values.tolist()

# window 슬라이싱
window_size =10
data_x, data_y = [], []
for i in range(len(y) - window_size):
    data_x.append(x[i : i + window_size])
    data_y.append(y[i + window_size])

# 훈련, 검증, 테스트 데이터 분할
train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[:train_size])
train_y = np.array(data_y[:train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])
test_x = np.array(data_x[train_size+val_size:])
test_y = np.array(data_y[train_size+val_size:])

print('Train:', train_x.shape, train_y.shape)
print('Val:', val_x.shape, val_y.shape)
print('Test:', test_x.shape, test_y.shape)

# 기존 LSTM 플랫 데이터 유지
model = Sequential()
model.add(LSTM(units =20, activation = 'tanh',
               return_sequences = True,
               input_shape = (10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units =20, activation = 'tanh'))
model.add(Dropout(0.1))
model.add(Dense(units = 1))
model.summary()

model.compile(optimizer = 'adam',
             loss = 'mean_squared_error')
history = model.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs = 70, batch_size = 30)

# --- 시간 보고서 구현을 위해 규가된 커드 ---

# 보정: RNN, GRU 모델 정의
model_rnn = Sequential()
model_rnn.add(SimpleRNN(20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
model_rnn.add(Dropout(0.1))
model_rnn.add(SimpleRNN(20, activation='tanh'))
model_rnn.add(Dropout(0.1))
model_rnn.add(Dense(1))
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
model_rnn.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)

model_gru = Sequential()
model_gru.add(GRU(20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
model_gru.add(Dropout(0.1))
model_gru.add(GRU(20, activation='tanh'))
model_gru.add(Dropout(0.1))
model_gru.add(Dense(1))
model_gru.compile(optimizer='adam', loss='mean_squared_error')
model_gru.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)

# 예측 값
pred_lstm = model.predict(test_x).flatten()
pred_rnn = model_rnn.predict(test_x).flatten()
pred_gru = model_gru.predict(test_x).flatten()
actual = np.array(test_y).flatten()


# 예측 수행
predicted = model.predict(test_x)

# 실제 값과 예측 값을 납작하게
actual = test_y.flatten()
predicted = predicted.flatten()


# 그래프
plt.figure(figsize=(8,6))
plt.plot(actual, label='Actual', color='red')
plt.plot(pred_rnn, label='predicted (rnn)', color='blue')
plt.plot(pred_gru, label='predicted (gru)', color='orange')
plt.plot(pred_lstm, label='predicted (lstm)', color='green')
plt.xlabel('time')
plt.ylabel('stock price')
plt.title('SEC stock price prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
