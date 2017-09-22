# Image 2 Vec with PyTorch

### Why do you want image embeddings?
 - Ranking for recommender systems
 - Clustering images to different categories
 - Classification tasks

## Available models
 - Resnet-18 (CPU, GPU)

## Installation

Tested on Python 3.6

#### Dependencies

Pytorch: http://pytorch.org/

Pillow:  ```pip install Pillow```

##### For running the example, you will additionally need:
 * Numpy: ```pip install numpy```
 * Sklearn ```pip install scikit-learn```

## Running the example
```git clone https://github.com/christiansafka/img2vec.git```

```cd img2vec/example```

```python test_img_to_vec.py```

#### Expected output
```
Which filename would you like similarities for?
cat.jpg
0.72832 cat2.jpg
0.641478 catdog.jpg
0.575845 face.jpg
0.516689 face2.jpg

Which filename would you like similarities for?
face2.jpg
0.668525 face.jpg
0.516689 cat.jpg
0.50084 cat2.jpg
0.484863 catdog.jpg
```
Try adding your own photos!

## Using img2vec as a library
Coming soon, but look at **test_img_to_vec.py** for an example 




