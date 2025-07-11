# data_utils.py
import FinanceDataReader as fdr
import numpy as np

# 날짜를 여기에서 한방에 관리
start_date = "2018-03-04"
end_date   = "2025-05-22"

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def load_samsung_data(window_size=20):
    df = fdr.DataReader('005930', start_date, end_date)
    dfx = df[['Open','High','Low','Volume','Close']]
    dfx = MinMaxScaler(dfx)
    dfy = dfx[['Close']]
    dfx = dfx[['Open','High','Low','Volume']]

    x = dfx.values.tolist()
    y = dfy.values.tolist()

    data_x, data_y = [], []
    for i in range(len(y) - window_size):
        data_x.append(x[i:i+window_size])
        data_y.append(y[i+window_size])

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    train_size = int(len(data_y) * 0.7)
    val_size   = int(len(data_y) * 0.2)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    val_x   = data_x[train_size:train_size+val_size]
    val_y   = data_y[train_size:train_size+val_size]
    test_x  = data_x[train_size+val_size:]
    test_y  = data_y[train_size+val_size:]

    return train_x, train_y, val_x, val_y, test_x, test_y
