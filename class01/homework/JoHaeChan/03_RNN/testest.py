import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout


def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2000-01-01', '2025-07-01')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

x = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 30
data_x, data_y = [], []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[:train_size])
train_y = np.array(data_y[:train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])
test_x = np.array(data_x[train_size+val_size:])
test_y = np.array(data_y[train_size+val_size:])

print('훈련 데이터:', train_x.shape, train_y.shape)
print('검증 데이터:', val_x.shape, val_y.shape)
print('테스트 데이터:', test_x.shape, test_y.shape)

model_rnn = Sequential([
    SimpleRNN(100, activation='tanh', return_sequences=True, input_shape=(window_size, 4)),
    Dropout(0.1),
    SimpleRNN(100, activation='tanh'),
    Dropout(0.1),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
rnn_history = model_rnn.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)

model_gru = Sequential([
    GRU(20, activation='tanh', return_sequences=True, input_shape=(window_size, 4)),
    Dropout(0.1),
    GRU(20, activation='tanh'),
    Dropout(0.1),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mean_squared_error')
gru_history = model_gru.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)

model_lstm = Sequential([
    LSTM(20, activation='tanh', return_sequences=True, input_shape=(window_size, 4)),
    Dropout(0.1),
    LSTM(20, activation='tanh'),
    Dropout(0.1),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
lstm_history = model_lstm.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['loss'], label='RNN Train')
plt.plot(gru_history.history['loss'], label='GRU Train')
plt.plot(lstm_history.history['loss'], label='LSTM Train')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(rnn_history.history['val_loss'], label='RNN Val')
plt.plot(gru_history.history['val_loss'], label='GRU Val')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
