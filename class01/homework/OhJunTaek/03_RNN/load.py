import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model

def load_data():
    with open("preprocessed_data.pkl", "rb") as f:
        return pickle.load(f)

def predict_and_plot(test_x, test_y):
    print("📊 모델 불러오기 및 예측 수행 중...")
    model_rnn = load_model("rnn_model.h5")
    model_gru = load_model("gru_model.h5")
    model_lstm = load_model("lstm_model.h5")

    pred_rnn = model_rnn.predict(test_x).flatten()
    pred_gru = model_gru.predict(test_x).flatten()
    pred_lstm = model_lstm.predict(test_x).flatten()
    test_y = test_y.flatten()

    time = np.arange(len(test_y))

    plt.figure(figsize=(10, 6))
    plt.plot(time, test_y, label="Actual", color='red')
    plt.plot(time, pred_rnn, label="Predicted (RNN)", color='blue')
    plt.plot(time, pred_gru, label="Predicted (GRU)", color='orange')
    plt.plot(time, pred_lstm, label="Predicted (LSTM)", color='green')
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("SEC stock price prediction (RNN / GRU / LSTM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_data()
    predict_and_plot(test_x, test_y)
