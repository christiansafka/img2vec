import sys
import os
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

samples = 5  # Amount of samples to take from input path
k_value = 2  # How many clusters

vec_mat = np.zeros((samples, vec_length))
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

for i in set(kmeans.labels_):
    try:
        os.mkdir('./' + str(i))
    except FileExistsError:
        continue

print('Predicting...')
preds = kmeans.predict(reduced_data)

print('Moving images...')
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    os.rename(input_path + '/' + filename, './' + str(preds[index]) + '/' + filename)

print('Done!')
