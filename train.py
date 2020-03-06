import os, sys
import numpy as np
import requests
from clint.textui import progress

input_sentences = []
output_sentences = []
output_sentences_inputs = []

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(BASE_DIR, 'data/eng-fra.txt')
GLOVE_DIR = os.path.join(BASE_DIR, 'data/glove.6B.100d.txt')

if os.path.isfile(GLOVE_DIR):
    print('Glove file exists')
else:
    print('Glove file does not exist, downloading ...')

    url = 'https://www.floydhub.com/api/v1/resources/Av2ThePYtAHXMAuSXEBV8X/glove.6B.100d.txt?content=true&rename=glove6b100dtxt'

    r = requests.get(url, stream=True)

    with open(os.path.join(BASE_DIR, 'data/glove.6B.100d.txt'), 'wb') as file:

        raw_content = r.raw.read()
        total_length = len(raw_content)

        for ch in progress.bar(r.iter_content(chunk_size=2391975), expected_size=(total_length / 1024) + 1):

            if ch:
                file.write(ch)

NUM_SENTENCES = 20000
count = 0
for line in open(DATADIR, encoding='utf-8'):
    count +=1
    if count > NUM_SENTENCES:
        break

    if '\t' not in line:
        continue

    input_sentence, output = line.rstrip().split('\t')

    output_sentence = output + ' <eos>'
    output_sentence_input = '<sos> ' + output

    input_sentences.append(input_sentence)
    output_sentences.append(output_sentence)
    output_sentences_inputs.append(output_sentence_input)

from tensorflow.keras.preprocessing.text import Tokenizer

MAX_NUM_WORDS = 20000


input_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index


max_input_len = max(len(sen) for sen in input_integer_seq)


output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index


num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)


from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)


# padding the decoder outputs
decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')

from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open(r'{}'.format(GLOVE_DIR), encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

EMBEDDING_SIZE = 100
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
for word, index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

from tensorflow.keras.layers import Embedding, Input, LSTM, Dense

# embedding layer for the input
embedding_layer = Embedding(num_words, EMBEDDING_SIZE, weights=[embedding_matrix], input_length=max_input_len)

decoder_targets_one_hot = np.zeros((
        len(input_sentences),
        max_out_len,
        num_words_output
    ),
    dtype='uint8'
)

decoder_targets_one_hot.shape

decoder_output_sequences = pad_sequences(output_integer_seq, maxlen=max_out_len, padding='post')
for i, d in enumerate(decoder_output_sequences):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

LSTM_NODES =256
encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

from tensorflow.keras.models import Model

model = Model([encoder_inputs_placeholder,
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

BATCH_SIZE = 64
EPOCHS = 20
r = model.fit(
    [encoder_input_sequences, decoder_input_sequences],
    decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
)

encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LSTM_NODES,))
decoder_state_input_c = Input(shape=(LSTM_NODES,))
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

encoder_model.save('models/encoder_model.h5')
decoder_model.save('models/decoder_model.h5')

