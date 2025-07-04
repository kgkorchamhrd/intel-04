import matplotlib.pyplot as plt
import tensorflow
import pickle

def gethistory(name):
    with open(f'{name}.history', 'rb') as f:
        history = pickle.load(f)
    return history

rnn_history = gethistory('rnn')   
gru_history = gethistory('gru')
lstm_history = gethistory('lstm')

print(rnn_history.keys(), gru_history.keys(), lstm_history.keys())

rnn_loss = rnn_history['loss']
rnn_val_loss = rnn_history['val_loss']
gru_loss = gru_history['loss']
gru_val_loss = gru_history['val_loss']
lstm_loss = lstm_history['loss']
lstm_val_loss = lstm_history['val_loss']

epochs = 70

epochs_range = range(epochs)
plt.figure(figsize=(12, 9))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, rnn_loss, label='RNN Training Loss')
plt.plot(epochs_range, gru_loss, label='GRU Training Loss')
plt.plot(epochs_range, lstm_loss, label='LSTM Training Loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, rnn_val_loss, label='RNN_Validation Loss')
plt.plot(epochs_range, gru_val_loss, label='GRU_Validation Loss')
plt.plot(epochs_range, lstm_val_loss, label='LSTM_Validation Loss')
plt.legend(loc='upper right')
plt.title('Validation Loss')
plt.show()
