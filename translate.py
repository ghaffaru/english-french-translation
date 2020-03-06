import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# print(input_sentences[100])
# from decoder_predict import decoder_model
# from encoder import encoder_model

encoder_model = load_model('models/encoder_model.h5')

decoder_model = load_model('models/decoder_model.h5')

NUM_SENTENCES = 20000
count = 0

input_sentences = []
output_sentences = []
output_sentences_inputs = []

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(BASE_DIR, 'data/eng-fra.txt')

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


input_tokenizer = Tokenizer(num_words=20000)
input_tokenizer.fit_on_texts(input_sentences)
input_integer_seq = input_tokenizer.texts_to_sequences(input_sentences)

word2idx_inputs = input_tokenizer.word_index
# print('Total unique words in the input: %s' % len(word2idx_inputs))

max_input_len = max(len(sen) for sen in input_integer_seq)
# print("Length of longest sentence in input: %g" % max_input_len)

output_tokenizer = Tokenizer(num_words=20000, filters='')
output_tokenizer.fit_on_texts(output_sentences + output_sentences_inputs)
output_integer_seq = output_tokenizer.texts_to_sequences(output_sentences)
output_input_integer_seq = output_tokenizer.texts_to_sequences(output_sentences_inputs)

word2idx_outputs = output_tokenizer.word_index
# print('Total unique words in the output: %s' % len(word2idx_outputs))

num_words_output = len(word2idx_outputs) + 1
max_out_len = max(len(sen) for sen in output_integer_seq)
# print("Length of longest sentence in the output: %g" % max_out_len)
idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

# print(word2idx_outputs)


def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

# print(max_input_length)
# sample_input_sentence = [input('Enter sentence to translate (Press q to quit) \n')]
# sample_input_tokenizer = Tokenizer(num_words=20000)
# sample_input_tokenizer.fit_on_texts(input_sentences)
#
# sample_input_integer_seq = sample_input_tokenizer.texts_to_sequences(sample_input_sentence)
# sample_max_input_len = max(len(sen) for sen in sample_input_integer_seq)
# sample_input_seq = pad_sequences(sample_input_integer_seq, maxlen=max_input_len)
# # print(sample_input_seq.shape)
# print(translate_sentence(sample_input_seq))
# input_sentence = [input('Enter a sentence (Press q to quit) \n')]
while input_sentence != ['q']:

        input_sentence = [input('Enter a sentence (Press q to quit) \n')]

        input_tokenizer = Tokenizer(num_words=20000)

        input_tokenizer.fit_on_texts(input_sentences)

        input_integer_sequence = input_tokenizer.texts_to_sequences(input_sentence)

        input_seqence = pad_sequences(input_integer_sequence, maxlen=max_input_len)

        print(translate_sentence(input_seqence))
quit()


