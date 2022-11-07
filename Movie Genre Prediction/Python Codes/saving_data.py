import csv
file = open('RNN_Output.csv', 'w', newline='')
writer = csv.writer(file) 
for item in preds:
    writer.writerow([item])


#with file:     
#    write.writerows(preds)

file.close()