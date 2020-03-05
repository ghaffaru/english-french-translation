from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tokenization import max_out_len,num_words_output
from encoder import encoder_states
# embedding_layer = Embedding(num_words, 100, weights=[embedding_matrix], input_length=max_input_length)
decoder_inputs_layer = Input(shape=(max_out_len))

decoder_embedding = Embedding(num_words_output, 256)

decoder_inputs_x = decoder_embedding(decoder_inputs_layer)

decoder_lstm = LSTM(256, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_inputs_single = Input(shape=(1,))

decoder_state_input_h = Input(
    shape=(256,)
)

decoder_state_input_c = Input(
    shape=(256,)
)

decoder_states_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [
    h,
    c
]

decoder_dense = Dense(
    num_words_output,
    activation='softmax'
)

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [
        decoder_inputs_single
    ] + decoder_states_inputs,
    [
        decoder_outputs
    ] + decoder_states
)

decoder_model.save('models/decoder_model.h5')