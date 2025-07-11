import FinanceDataReader as fdr
import numpy as np
import pickle
import os

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def preprocess_and_save(window_size=20):
    df = fdr.DataReader('005380', '2024-06-22', '2025-06-22')
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

if __name__ == "__main__":
    preprocess_and_save()  # 조건 없이 항상 새로 저장

