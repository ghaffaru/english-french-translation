from tokenization import input_integer_sequences, output_input_integer_seq, max_input_length, max_out_len,output_integer_seq

from tensorflow.keras.preprocessing.sequence import pad_sequences

encoder_input_sequences = pad_sequences(input_integer_sequences, maxlen=max_input_length)

# print(encoder_input_sequences.shape)
decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')

decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
# print(decoder_input_sequences.shape)
