import load_data

from tensorflow.keras.preprocessing.text import Tokenizer

MAX_NUM_WORDS = 20000
input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

input_tokenizer.fit_on_texts(load_data.input_sentences)

input_integer_sequences = input_tokenizer.texts_to_sequences(load_data.input_sentences)

word2index_inputs = input_tokenizer.word_index

# print(len(word2index_inputs))

max_input_length = max(len(sentence) for sentence in input_integer_sequences)

print(max_input_length);