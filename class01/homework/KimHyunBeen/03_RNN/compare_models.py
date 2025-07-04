from tensorflow.keras.models import load_model
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_samsung_data

# 테스트 데이터 로드
_, _, _, _, test_x, test_y = load_samsung_data()

# 모델 불러오기
model_rnn  = load_model('./03_RNN/rnn_model.keras')
model_gru  = load_model('./03_RNN/gru_model.keras')
model_lstm = load_model('./03_RNN/lstm_model.keras')

# 예측
pred_rnn  = model_rnn.predict(test_x)
pred_gru  = model_gru.predict(test_x)
pred_lstm = model_lstm.predict(test_x)

# 그래프
plt.figure(figsize=(14,8))
plt.plot(test_y, label="Real")
plt.plot(pred_rnn, label="RNN")
plt.plot(pred_gru, label="GRU")
plt.plot(pred_lstm, label="LSTM")
plt.legend()
plt.title("RNN / GRU / LSTM 모델 예측 비교")
plt.show()
