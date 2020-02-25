NUM_SENTENCES = 20000

input_sentences = [] # sentences in the original language as input to the encoder

output_sentences = [] # actual target sentence with an end-of-sentence token

output_sentences_inputs = [] # input to the decoder in the translated language

count = 0

for line in open('/home/ghaff/Artificial Intelligence/nlp-projects/english-french-translation/data/english.txt', encoding='utf-8'):
    count +=1

    if count > NUM_SENTENCES:
        break

    input_sentence = line.rstrip()

    input_sentences.append(input_sentence)

print(len(input_sentences))

count1 = 0
for line in open('/home/ghaff/Artificial Intelligence/nlp-projects/english-french-translation/data/french.txt', encoding='utf-8'):
    count1 +=1

    if count1 > NUM_SENTENCES:
        break
    output = line.rstrip()

    output_sentence = output + '<eos>'

    output_sentence_input = '<sos>' + output

    output_sentences.append(output_sentence)

    output_sentences_inputs.append(output_sentence_input)

print(len(output_sentences))




