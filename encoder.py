from tensorflow.keras.layers import Input, Embedding, LSTM
from tensorflow.keras.models import Model

from tokenization import max_input_length

from embedding import num_words, embedding_matrix

# The input to the encoder will be the sentence in English and the output will be the hidden state and cell state of the LSTM.

embedding_layer = Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_input_length)

encoder_inputs_layer = Input(shape=(max_input_length,))

x = embedding_layer(encoder_inputs_layer)

encoder = LSTM(256, return_state=True)

encoder_outputs, h, c = encoder(x)

encoder_states = [h, c]

encoder_model = Model(
    encoder_inputs_layer,
    encoder_states
)

# encoder_model.save('models/encoder_model.h5')