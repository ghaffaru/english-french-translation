from padding import encoder_input_sequences, decoder_input_sequences
from output_array import decoder_targets_one_hot
from net_architecture import model
from tensorflow.keras.models import Model
from encoder import encoder_inputs_layer, encoder_states
from decoder import decoder_embedding, decoder_lstm, decoder_dense
from tensorflow.keras.layers import Input
model.fit(
    [
        encoder_input_sequences,
        decoder_input_sequences
    ],
    decoder_targets_one_hot,
    batch_size=128,
    epochs=2,
    validation_split=0.1
)

model.save('models/model.h5')

encoder_model = Model(encoder_inputs_layer, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# from encoder import encoder_model
# from decoder_predict import decoder_model
#
encoder_model.save('models/encoder_model.h5')
decoder_model.save('models/decoder_model.h5')


