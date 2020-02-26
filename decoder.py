from tensorflow.keras.layers import Input, Embedding, LSTM

from tokenization import max_out_len
from encoder import encoder_states
# embedding_layer = Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_input_length)
decoder_inputs_layer = Input(shape=(max_out_len))

decoder_embedding = Embedding(num_words_output, 256)

decoder_inputs_x = decoder_embedding(decoder_inputs_layer)

decoder = LSTM(256, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder(decoder_inputs_x, initial_state=encoder_states)

