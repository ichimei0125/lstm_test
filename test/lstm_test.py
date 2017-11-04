import tensorflow as tf

number_of_layers = 1
lstm_size = 1
batch_size = 2
words_in_dataset = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=tf.float32)

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)

# Initial state of the LSTM memory.
state = tf.zeros([batch_size, lstm.state_size])

print(words_in_dataset)
print(state)

output, state = lstm(words_in_dataset, state)

