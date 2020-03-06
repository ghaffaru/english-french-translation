import load_data

from tensorflow.keras.preprocessing.text import Tokenizer

MAX_NUM_WORDS = 20000
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

input_tokenizer.fit_on_texts(load_data.input_sentences)

input_integer_sequences = input_tokenizer.texts_to_sequences(load_data.input_sentences)

word2index_inputs = input_tokenizer.word_index

# print(len(word2index_inputs))

max_input_length = max(len(sentence) for sentence in input_integer_sequences)

# print(max_input_length)

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')

output_tokenizer.fit_on_texts(load_data.output_sentences + load_data.output_sentences_inputs)

output_integer_seq = output_tokenizer.texts_to_sequences(load_data.output_sentences)

output_input_integer_seq = output_tokenizer.texts_to_sequences(load_data.output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index

num_words_output = len(word2idx_outputs) + 1
# print('Total unique words in the output: %s' % len(word2idx_outputs))
# print(word2idx_outputs)

max_out_len = max(len(sen) for sen in output_integer_seq)

# print("Length of longest sentence in the output: %g" % max_out_len)