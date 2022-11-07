import os
import copy
import numpy as np
import pandas as pd
from PIL import Image
from time import sleep
from itertools import chain
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms

resize_transform = transforms.Resize((300, 300))
rotate_transform = transforms.RandomRotation(60)

X = []
y = []

parent_path = r'C:\Users\Atiyeh\Desktop\train_images\train_set' 
aug_data_path = r'C:\Users\Atiyeh\Desktop\ML Final Project\train_images\train_set'
data = pd.read_csv(r'C:\Users\Atiyeh\Desktop\ML Final Project\train_data\train_set.csv', dtype=str)

address = data['image']
genres = data['genre']

genre = data['genre'].astype(dtype="category")
aa = data.groupby(genre)
cats = list(genre.cat.categories)

num_cat = list(data.groupby('genre').count()['image'])
cat_count_dict = dict(zip(cats, num_cat))
print(cat_count_dict)

num_target = 1100
new_data = pd.DataFrame(columns=data.columns)
cat_count_dict = dict()
categories_num = {key: 0 for key in cats} 

c = 0

while True:

    for idx, (row, local_path, genre) in enumerate(zip(data.iterrows(), address, genres)):
        abs_path = os.path.join(parent_path, local_path)
        img = Image.open(abs_path)

        if categories_num[genre]==1099:
            c += 1
            img_name = genre + str(c) + ".jpg"
            img_path = os.path.join(aug_data_path, img_name)
            img.save(img_path)
            
            row_new = copy.deepcopy(row)
            row_new[1]['image'] = img_name
            new_data = new_data.append(row_new[1], ignore_index=True)
            categories_num[genre] += 1
        
        elif categories_num[genre]<1099:
            c += 1
            img_name = genre + str(c) + ".jpg"
            img_path = os.path.join(aug_data_path, img_name)
            img.save(img_path)
            row_new = copy.deepcopy(row)
            row_new[1]['image'] = img_name
            new_data = new_data.append(row_new[1], ignore_index=True)
            categories_num[genre] += 1
            
            c += 1
            resized_img = resize_transform(img)
            transformed_img = rotate_transform(resized_img)
            transformed_img_name = genre + str(c) + ".jpg"
            img_path = os.path.join(aug_data_path, transformed_img_name)
            transformed_img.save(img_path)
            row_new = copy.deepcopy(row)
            row_new[1]['image'] = transformed_img_name
            new_data = new_data.append(row_new[1], ignore_index=True)
            categories_num[genre] += 1
        
        if all(np.array(list(new_data.groupby('genre').count()['image'])) == 1100):
            break
    if all(np.array(list(new_data.groupby('genre').count()['image'])) == 1100):
        break




    