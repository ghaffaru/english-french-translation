import numpy as np
from load_data import input_sentences
from tokenization import max_out_len, num_words_output
from padding import decoder_output_sequences

decoder_targets_one_hot = np.zeros(
    (
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='uint8'
)
# print(decoder_targets_one_hot.shape)
for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

