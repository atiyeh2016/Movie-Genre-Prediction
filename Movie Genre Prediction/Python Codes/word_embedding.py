import pandas as pd
from hazm import word_tokenize
from gensim.models.word2vec import Word2Vec

df = pd.read_csv('dataset.csv', encoding="utf-8")
descriptions = df.description_fa
descriptions.fillna("", inplace=True)
descriptions = [description for description in descriptions]

sents = [word_tokenize(description) for description in descriptions]


model = Word2Vec(sentences=sents, size=64, window=10, min_count=5, seed=42, workers=5)
model.save('model.bin')

new_model = Word2Vec.load('model.bin')
print(new_model)