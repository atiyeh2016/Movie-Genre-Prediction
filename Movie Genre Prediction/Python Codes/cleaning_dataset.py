from PreprocessingDescriptions import Preprocessing
import pathlib
import csv
import os

#%% data preprocessing --> train

P = Preprocessing()
path = pathlib.Path().absolute()
path = os.path.join(path, "dataset.csv")
with open(path, encoding="utf-8") as csvfile:
    data_reader = csv.reader(csvfile)
    next(data_reader, None)
    for row in data_reader:
        genre = row[8]
        description = row[3]
        P.preprocess_descriptions(description, genre)
       
#%% preprocessing and making new csv --> train

with open('dataset.cleaned.csv', 'w', newline='', encoding="utf-8") as csvfile:
    fieldnames = ['descriptions','genres']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for cleaned_text, genre in zip(P.desc, P.genre):
        cleaned_text = {'genres': genre ,'descriptions': cleaned_text}
        writer.writerow(cleaned_text)