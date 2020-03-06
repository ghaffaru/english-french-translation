import numpy as np
from tokenization import MAX_NUM_WORDS, word2index_inputs
import os

import requests
from clint.textui import progress

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
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

                for ch in progress.bar(r.iter_content(chunk_size = 2391975), expected_size=(total_length/1024) + 1):
    
                        if ch:

                                file.write(ch)


EMBEDDING_SIZE = 100
# dictionary where words are the keys and the corresponding vectors are values
embeddings_dict = dict()

glove_file = open(GLOVE_DIR, encoding='utf8')

for line in glove_file:
    records = line.split()

    word = records[0]

    vector_dimensions = np.asarray(records[1:], dtype='float32')

    embeddings_dict[word] = vector_dimensions

glove_file.close()


# matrix that will contain the word embeddings for the words in our input sentences
num_words = min(MAX_NUM_WORDS, len(word2index_inputs) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_SIZE))

for word, index in word2index_inputs.items():
    embedding_vector = embeddings_dict.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector

# print(embeddings_dict['hello'])

# This word embedding matrix will be used to create the embedding layer for the LSTM model.