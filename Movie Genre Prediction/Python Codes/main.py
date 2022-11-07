from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
#from preprocessing import Preprocessing
from FilmsDataset import FilmsDataset
from torchvision import transforms
#from CenterCrop import CenterCrop
from collections import Counter
import matplotlib.pyplot as plt 
from PIL import Image
import scipy.io as io
import torch.nn as nn
import pandas as pd
import numpy as np
import pathlib
import torch
import os


#def makingDataFrame(genre, img_address):
#    return pd.concat([img_address, genre], axis=1)

#parent_path = r'C:\Users\Atiyeh\Desktop\ML Final Project\train_images\train_set'  
#data = pd.read_csv(r'C:\Users\Atiyeh\Desktop\ML Final Project\train_data\train_set.csv')
#genre = data['genre'].astype(dtype="category")
#img_address = data['image']
##print(genre.cat.categories)
#labels = genre.cat.codes
##print(labels)
#print(data.groupby('genre').count())

#genre_and_imgnames = makingDataFrame(labels, img_address)
#genre_and_imgnames.to_csv('imgnames_and_genres.csv', index=False)
fig, axs = plt.subplots(1,2, figsize=(8, 8))

parent_path = pathlib.Path().absolute().parent
folder1 = 'train_images'
folder2 = 'train_set'
parent = os.path.join(parent_path, folder1, folder2)

resize_transform = transforms.Resize((350,300))
rotate_transform = transforms.RandomRotation(60)

img = r'action865.jpg'
img_path = os.path.join(parent, img)
img = Image.open(img_path)
axs[0].imshow(img)
transformed_img = rotate_transform(img)
axs[1].imshow(transformed_img)


#transforms = transforms.Compose([
#                                 transforms.Resize((380,330)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0,0,0), (1,1,1))
#                                ])
#
#transformed_dataset = FilmsDataset('imgnames_and_genres.csv', img_path,
#                                   transform=transforms)
#
#dataloader = DataLoader(transformed_dataset, batch_size=4,
#                        shuffle=True, num_workers=0)
#
#sample_image = transformed_dataset[5823]

#data = pd.read_csv(r'C:\Users\Atiyeh\Desktop\ML Final Project\train_data\train_set.csv')
#print(data.groupby('genre').count())
#a = data.groupby('genre').count()
#print(np.mean(a['image']))

#oversample = SMOTE()
#X, y = oversample.fit_resample(X, y)
#
#counter = Counter(y)
#counter = Counter(y)





