import pandas as pd
import matplotlib.pyplot as plt
import csv

descs_len = []
with open('dataset.cleaned.csv', encoding="utf-8") as csvfile:
    descriptions = csv.reader(csvfile)
    for description in descriptions:
        descs_len.append(len(description[0]))
        
pd.Series(descs_len).hist()
plt.show()
print(pd.Series(descs_len).describe())