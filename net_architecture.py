from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from tokenization import num_words_output
from decoder import decoder_outputs

from encoder import encoder_inputs_layer
from decoder import decoder_inputs_layer

densed_decoder = Dense(num_words_output, activation='softmax')

decoder_outputs = densed_decoder(decoder_outputs)

# init the model

model = Model(
    [
        encoder_inputs_layer,
        decoder_inputs_layer,
    ],
    decoder_outputs
)

# compile model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)