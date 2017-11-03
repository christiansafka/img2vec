import sys
import os
from shutil import copyfile
sys.path.append("..")  # Adds higher directory to python modules path.

from img_to_vec import Img2Vec
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

input_path = './test_images'
files = os.listdir(input_path)

img2vec = Img2Vec()
vec_length = 512  # Using resnet-18 as default

samples = len(files)  # Amount of samples to take from input path
k_value = 2  # How many clusters

# Matrix to hold the image vectors
vec_mat = np.zeros((samples, vec_length))
# If samples is < the number of files in the folder, we sample them randomly
sample_indices = np.random.choice(range(0, len(files)), size=samples, replace=False)

print('Reading images...')
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    vec_mat[index, :] = vec

print('Applying PCA...')
reduced_data = PCA(n_components=2).fit_transform(vec_mat)
kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10)
kmeans.fit(reduced_data)

# Create a folder for each cluster (0, 1, 2, ..)
for i in set(kmeans.labels_):
    try:
        os.mkdir('./' + str(i))
    except FileExistsError:
        continue

print('Predicting...')
preds = kmeans.predict(reduced_data)

print('Copying images...')
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    copyfile(input_path + '/' + filename, './' + str(preds[index]) + '/' + filename)

print('Done!')
