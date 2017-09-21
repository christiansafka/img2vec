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
0.791242 cat2.jpg
0.677612 face.jpg
0.656266 catdog.jpg
0.585546 face2.jpg

Which filename would you like similarities for?
face2.jpg
0.763177 face.jpg
0.585546 cat.jpg
0.531366 cat2.jpg
0.489123 catdog.jpg
```
Try adding your own photos!

## Using img2vec as a library
Coming soon, but look at **test_img_to_vec.py** for an example 




