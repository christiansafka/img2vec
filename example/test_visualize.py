import sys
import os
sys.path.append("..")  # Adds higher directory to python modules path.
from img_to_vec import Img2Vec
from PIL import Image
from sklearn.manifold import TSNE
import numpy as np
from ggplot import *
import pandas as pd

input_path = './test_images'
files = os.listdir(input_path)

img2vec = Img2Vec()
vec_length = 512  # Using resnet-18 as default

samples = 50  # Amount of samples to take from input path

vec_mat = np.zeros((samples, vec_length))
sample_indices = np.random.choice(range(0, len(files)), size=samples, replace=False)

# For each test sample, we store the label and append the img vector to our matrix
labels = []
for index, i in enumerate(sample_indices):
    file = files[i]
    filename = os.fsdecode(file)
    img = Image.open(os.path.join(input_path, filename))
    vec = img2vec.get_vec(img)
    vec_mat[index, :] = vec
    if 'dog' in filename:
        labels.append('dog')
    elif 'cat' in filename:
        labels.append('cat')
    elif 'face' in filename:
        labels.append('face')

t_sne = TSNE()

tsne_output = t_sne.fit_transform(vec_mat)

df_tsne = pd.DataFrame(columns=['x-tsne', 'y-tsne', 'label'])
df_tsne['x-tsne'] = tsne_output[:, 0]
df_tsne['y-tsne'] = tsne_output[:, 1]
df_tsne['label'] = labels

chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
        + geom_point(size=90, alpha=0.8) \
        + ggtitle("tSNE dimensions colored by image label")

print(chart)
