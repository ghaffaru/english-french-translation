from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tokenization import num_words_output

decoder_embedding = Embedding(num_words_output, 256)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))

decoder_states_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

decoder_inputs_single = Input(shape=(1,))

decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]

decoder_dense = Dense(
    num_words_output,
    activation='softmax'
)

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

decoder_model.save('models/decoder_model.h5')