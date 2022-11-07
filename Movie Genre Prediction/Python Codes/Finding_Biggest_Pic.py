import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import random

parent_path = r'C:\Users\Atiyeh\Desktop\ML Final Project\train_images\train_set'

fig, axs = plt.subplots(1,3, figsize=(10,10))
#fig.suptitle('Some Sample Posters')
entries = os.listdir(r'C:\Users\Atiyeh\Desktop\ML Final Project\train_images\train_set')

post1 = random.randint(1,11000)
new_path = os.path.join(parent_path, entries[post1])
img = plt.imread(new_path)
axs[0].imshow(img)

post2 = random.randint(1,11000)
new_path = os.path.join(parent_path, entries[post2])
img = plt.imread(new_path)
axs[1].imshow(img)

post3 = random.randint(1,11000)
new_path = os.path.join(parent_path, entries[post3])
img = plt.imread(new_path)
axs[2].imshow(img)
plt.show()

print(img.shape)

all_h = []
all_w = []

max_h = 0
max_w = 0

min_h = 500
min_w = 500

mean_h = 0
mean_w = 0

for idx, entry in enumerate(entries):
    new_path = os.path.join(parent_path, entry)
    img = plt.imread(new_path)
    
    h = img.shape[0]
    w = img.shape[1]
    
    if h>max_h:
        max_h = h
    if w>max_w:
        max_w = w
        
    if h<min_h:
        min_h = h
    if w<min_w:
        min_w = w
        
    mean_h = (mean_h*(idx)+h)/(idx+1)
    mean_w = (mean_w*(idx)+w)/(idx+1)
    
    all_h.append(h)
    all_w.append(w)
    

number_of_bins = 15
#number_of_bins = 'auto'

plt.figure()
plt.suptitle('Images Height Histogram')
hist_h = plt.hist(all_h, bins = number_of_bins)
plt.show()
#plt.suptitle('Images Height Histogram')

plt.figure()
plt.suptitle('Images Width Histogram')
hist_w = plt.hist(all_w, bins = number_of_bins)
plt.show()
