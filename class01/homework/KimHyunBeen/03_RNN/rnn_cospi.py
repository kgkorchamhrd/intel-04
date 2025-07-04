import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from data_utils import load_samsung_data
import os


# 데이터 로드
train_x, train_y, val_x, val_y, test_x, test_y = load_samsung_data()

os.makedirs("./03_RNN", exist_ok=True)



model = Sequential()
model.add(SimpleRNN(units=20, activation='tanh',
                    return_sequences=True,
                    input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=20, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=1000, batch_size=10)
model.save('./03_RNN/rnn_model.keras')



model = Sequential()
model.add(GRU(units=20, activation='tanh',
              return_sequences=True,
              input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(GRU(units=20, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=1000, batch_size=10)
model.save('./03_RNN/gru_model.keras')




model = Sequential()
model.add(LSTM(units=20, activation='tanh',
               return_sequences=True,
               input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error')
history = model.fit(train_x, train_y,
                    validation_data = (val_x, val_y),
                    epochs=1000, batch_size=10)
model.save('./03_RNN/lstm_model.keras')