NUM_SENTENCES = 20000

input_sentences = [] # sentences in the original language as input to the encoder

output_sentences = [] # actual target sentence with an end-of-sentence token

output_sentences_inputs = [] # input to the decoder in the translated language

count = 0

for line in open('/home/ghaff/Artificial Intelligence/nlp-projects/eng-french-translation/data/english.txt', encoding='utf-8'):
    count +=1

    if count > NUM_SENTENCES:
        break

    input_sentence = line.rstrip()

    input_sentences.append(input_sentence)

print(len(input_sentences))





