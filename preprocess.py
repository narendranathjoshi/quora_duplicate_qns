import gensim
import csv
import json
import os

sentences_path = 'proc/sentences.json'
model_path = 'proc/qns_word2vec.bin'

def iterctr(items, n):
    ctr = 0
    for item in items:
        ctr += 1
        if ctr % n == 0:
            print(ctr)

        yield item

sentences = []

if not os.path.exists(sentences_path):
    print('Sentences not found. Constructing ...')
    with open('data/train.csv', 'rt') as f:
        reader = csv.DictReader(f)
        for row in iterctr(reader, 10000):
            sentences.append(list(gensim.utils.tokenize(row['question1'], lowercase=True)))
            sentences.append(list(gensim.utils.tokenize(row['question2'], lowercase=True)))

    with open(sentences_path, 'w') as outfile:
        json.dump(sentences, outfile)
else:
    print('Sentences found. Constructing ...')
    with open(sentences_path) as data_file:
        sentences = json.load(data_file)

print('Sentences constructed')

if not os.path.exists(model_path):
    print('Model not found. Training and saving ...')
    model = gensim.models.Word2Vec(sentences, min_count=1)
    model.save(model_path)

print('Model trained and saved')