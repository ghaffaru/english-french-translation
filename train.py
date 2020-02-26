from padding import encoder_input_sequences, decoder_input_sequences

from net_architecture import model

model.fit(
    [
        encoder_input_sequences,
        decoder_input_sequences
    ],
    
)