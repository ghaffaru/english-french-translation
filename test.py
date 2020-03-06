import numpy as np
from load_data import input_sentences
from padding import encoder_input_sequences
from translate import translate_sentence

i = np.random.choice(len(input_sentences))
input_seq = encoder_input_sequences[i:i+1]
translation = translate_sentence(input_seq)
# print('-')
# print('Input:', input_sentences[i])
# print('Response:', translation)
# input_seq.shape
print(input_sentences[i])
print(translation)