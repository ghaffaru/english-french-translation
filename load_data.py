import os

# import urllib.request

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ENGLISH_PATH = os.path.join(BASE_DIR, 'data/english.txt')
FRENCH_PATH = os.path.join(BASE_DIR, 'data/french.txt')

# url = 'http://i3.ytimg.com/vi/J---aiyznGQ/mqdefault.jpg'
#
# urllib.request.urlretrieve(url, os.path.join(BASE_DIR, 'data/image.jpg'))

NUM_SENTENCES = 20000

input_sentences = []  # sentences in the original language as input to the encoder

output_sentences = []  # actual target sentence with an end-of-sentence token

output_sentences_inputs = []  # input to the decoder in the translated language

count = 0

for line in open(ENGLISH_PATH, encoding='utf-8'):
    count += 1

    if count > NUM_SENTENCES:
        break

    input_sentence = line.rstrip()

    input_sentences.append(input_sentence)

# print(len(input_sentences))

count1 = 0
for line in open(FRENCH_PATH, encoding='utf-8'):
    count1 += 1

    if count1 > NUM_SENTENCES:
        break
    output = line.rstrip()

    output_sentence = output + '<eos>'

    output_sentence_input = '<sos>' + output

    output_sentences.append(output_sentence)

    output_sentences_inputs.append(output_sentence_input)

# print(len(output_sentences))
