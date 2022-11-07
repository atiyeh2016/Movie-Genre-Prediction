#%% Importing 
from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from gensim.models.word2vec import Word2Vec
import pickle

embeddings_dict = Word2Vec.load('model.bin')
zero_vector = embeddings_dict["."]

mean_file = open('mean_vector.pkl', 'rb')
mean_vector = pickle.load(mean_file)

#%% Class Defenition
class FilmDescriptionsDataset(Dataset):

    def __init__(self, csv_file, desc_len = 250, transform=None):

        data = pd.read_csv(csv_file, encoding="utf-8")
        selected_columns = data[["descriptions","genres"]]
        self.descriptions = selected_columns.copy()
        self.descriptions['genres'] = self.descriptions['genres'].astype(dtype="category").cat.codes
        self.transform = transform
        self.desc_len = desc_len
        
    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        genre = self.descriptions.iloc[idx, 1]
        description = self.descriptions.iloc[idx, 0]
        try:
            vocabs = description.split()
        except AttributeError:
            vocabs = []
    
        vectors = []
        for vocab in vocabs:
            try:
                vectors.append(embeddings_dict[vocab.lower()])
            except:
                vectors.append(mean_vector)
            
        if len(vocabs)<self.desc_len:
            for n in range(self.desc_len-len(vocabs)):
                vectors.insert(0, zero_vector)
        elif len(vocabs)>self.desc_len:
             for n in range(len(vocabs)-self.desc_len):
                vectors.pop()
                
        sample = {'description': np.array(vectors), 'genre': genre}

        if self.transform:
            sample = self.transform(sample)
            

        return sample