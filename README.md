# Image 2 Vec with PyTorch

Medium post on building the first version from scratch:  https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

Looking for a simpler image vector integration for your project?  Check out our free API at https://latentvector.space

### Applications of image embeddings:
 - Ranking for recommender systems
 - Clustering images to different categories
 - Classification tasks
 - Image compression

## Available models
|Model name|Return vector length|
|----|----|
|Resnet-18|512|
|Alexnet|4096|
|Vgg-11|4096|
|Densenet|1024|
|efficientnet_b0|1280|
|efficientnet_b1|1280|
|efficientnet_b2|1408|
|efficientnet_b3|1536|
|efficientnet_b4|1792|
|efficientnet_b5|2048|
|efficientnet_b6|2304|
|efficientnet_b7|2560|

## Installation

Tested on Python 3.6 and torchvision 0.11.0 (nightly, 2021-09-25) 

Requires Pytorch: http://pytorch.org/

```conda install -c pytorch-nightly torchvision```

```pip install img2vec_pytorch```

## Run test

```python -m img2vec_pytorch.test_img_to_vec```

## Using img2vec as a library
```python
from img2vec_pytorch import Img2Vec
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)

# Read in an image (rgb format)
img = Image.open('test.jpg')
# Get a vector from img2vec, returned as a torch FloatTensor
vec = img2vec.get_vec(img, tensor=True)
# Or submit a list
vectors = img2vec.get_vec(list_of_PIL_images)
```

##### For running the example, you will additionally need:
 * Pillow:  ```pip install Pillow```
 * Sklearn ```pip install scikit-learn```

## Running the example
```git clone https://github.com/christiansafka/img2vec.git```

```cd img2vec/example```

```python test_img_similarity.py```

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


#### Img2Vec Params
**cuda** = (True, False) &nbsp; # Run on GPU? &nbsp; &nbsp; default: False<br>
**model** = ('resnet-18', 'alexnet', 'vgg', 'densenet') &nbsp; # Which model to use? &nbsp; &nbsp; default: 'resnet-18'<br>

## Advanced users 
----

### Read only file systems

If you use this library from the app running in read only environment (for example, docker container), 
specify writable directory where app can store pre-trained models. 

```bash
export TORCH_HOME=/tmp/torch
```

### Additional Parameters

**layer** = 'layer_name' or int &nbsp; # For advanced users, which layer of the model to extract the output from.&nbsp;&nbsp; default: 'avgpool' <br>
**layer_output_size** = int &nbsp; # Size of the output of your selected layer

### [Resnet-18](http://pytorch-zh.readthedocs.io/en/latest/_modules/torchvision/models/resnet.html)
Defaults: (layer = 'avgpool', layer_output_size = 512)<br>
Layer parameter must be an string representing the name of a  layer below
```python
conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
bn1 = nn.BatchNorm2d(64)
relu = nn.ReLU(inplace=True)
maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
layer1 = self._make_layer(block, 64, layers[0])
layer2 = self._make_layer(block, 128, layers[1], stride=2)
layer3 = self._make_layer(block, 256, layers[2], stride=2)
layer4 = self._make_layer(block, 512, layers[3], stride=2)
avgpool = nn.AvgPool2d(7)
fc = nn.Linear(512 * block.expansion, num_classes)
```
### [Alexnet](http://pytorch-zh.readthedocs.io/en/latest/_modules/torchvision/models/alexnet.html)
Defaults: (layer = 2, layer_output_size = 4096)<br>
Layer parameter must be an integer representing one of the layers below
```python
alexnet.classifier = nn.Sequential(
            7. nn.Dropout(),                  < - output_size = 9216
            6. nn.Linear(256 * 6 * 6, 4096),  < - output_size = 4096
            5. nn.ReLU(inplace=True),         < - output_size = 4096
            4. nn.Dropout(),		      < - output_size = 4096
            3. nn.Linear(4096, 4096),	      < - output_size = 4096
            2. nn.ReLU(inplace=True),         < - output_size = 4096
            1. nn.Linear(4096, num_classes),  < - output_size = 4096
        )
```

### [Vgg](https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html)
Defaults: (layer = 2, layer_output_size = 4096)<br>
```python
vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
```

### [Densenet](https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html)
Defaults: (layer = 1 from features, layer_output_size = 1024)<br>
```python
densenet.features = nn.Sequential(OrderedDict([
	('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
						padding=3, bias=False)),
	('norm0', nn.BatchNorm2d(num_init_features)),
	('relu0', nn.ReLU(inplace=True)),
	('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
]))
```

### [EfficientNet](https://arxiv.org/abs/1905.11946)
Defaults: (layer = 1 from features, layer_output_size = 1280 for efficientnet_b0 model)<br>


## To-do
- Benchmark speed and accuracy
- Add ability to fine-tune on input data
- Export documentation to a normal place




