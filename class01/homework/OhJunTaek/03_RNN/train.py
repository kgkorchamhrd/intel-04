import FinanceDataReader as fdr
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def preprocess_and_save(window_size=10):
    df = fdr.DataReader('005380 ', '2024-06-022', '2025-06-22')
    dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
    dfx = MinMaxScaler(dfx)
    dfy = dfx[['Close']]
    dfx = dfx[['Open', 'High', 'Low', 'Volume']]

    x = dfx.values.tolist()
    y = dfy.values.tolist()

    data_x, data_y = [], []
    for i in range(len(y) - window_size):
        data_x.append(x[i:i + window_size])
        data_y.append(y[i + window_size])

    train_size = int(len(data_y) * 0.7)
    val_size = int(len(data_y) * 0.2)

    train_x = np.array(data_x[0:train_size])
    train_y = np.array(data_y[0:train_size])
    val_x = np.array(data_x[train_size:train_size+val_size])
    val_y = np.array(data_y[train_size:train_size+val_size])
    test_x = np.array(data_x[train_size+val_size:])
    test_y = np.array(data_y[train_size+val_size:])

    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump((train_x, train_y, val_x, val_y, test_x, test_y), f)
    print("✅ 데이터 전처리 완료 및 저장됨")

def build_model(model_type, input_shape=(10,4)):
    model = Sequential()
    if model_type == 'rnn':
        model.add(SimpleRNN(20, activation='tanh', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(SimpleRNN(20, activation='tanh'))
    elif model_type == 'gru':
        model.add(GRU(20, activation='tanh', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(GRU(20, activation='tanh'))
    elif model_type == 'lstm':
        model.add(LSTM(20, activation='tanh', return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.1))
        model.add(LSTM(20, activation='tanh'))

    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save_models(train_x, train_y, val_x, val_y):
    for model_type in ['rnn', 'gru', 'lstm']:
        model = build_model(model_type)
        print(f"🔧 {model_type.upper()} 모델 학습 중...")
        es = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(train_x, train_y, validation_data=(val_x, val_y),
                  epochs=70, batch_size=30, callbacks=[], verbose=1)
        model.save(f"{model_type}_model.h5")
        print(f"💾 {model_type.upper()} 모델 저장됨 → {model_type}_model.h5")

if __name__ == "__main__":
    if not os.path.exists("preprocessed_data.pkl"):
        preprocess_and_save()
    train_x, train_y, val_x, val_y, test_x, test_y = None, None, None, None, None, None
    with open("preprocessed_data.pkl", "rb") as f:
        train_x, train_y, val_x, val_y, test_x, test_y = pickle.load(f)
    train_and_save_models(train_x, train_y, val_x, val_y)
