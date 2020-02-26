from padding import encoder_input_sequences, decoder_input_sequences
from output_array import decoder_targets_one_hot
from net_architecture import model

model.fit(
    [
        encoder_input_sequences,
        decoder_input_sequences
    ],
    decoder_targets_one_hot,
    batch_size=64,
    epochs=2,
    validation_split=0.1
)