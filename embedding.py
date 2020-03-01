import numpy as np
from tokenization import MAX_NUM_WORDS, word2index_inputs


EMBEDDING_SIZE = 100
# dictionary where words are the keys and the corresponding vectors are values
embeddings_dict = dict()

glove_file = open(r'/home/ghaff/Artificial Intelligence/nlp-projects/english-french-translation/data/glove.6B.100d.txt')

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