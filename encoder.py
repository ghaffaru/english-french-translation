from tensorflow.keras.layers import Input, Embedding

from tokenization import max_input_length
from embedding import num_words
# The input to the encoder will be the sentence in English and the output will be the hidden state and cell state of the LSTM.
embedding_layer = Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_input_length)

encoder = Input(shape=(max_input_length,))

x = embedding_layer(encoder)




