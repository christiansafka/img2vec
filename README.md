# Image 2 Vec with PyTorch

### Applications of image embeddings:
 - Ranking for recommender systems
 - Clustering images to different categories
 - Classification tasks

## Available models
 - Resnet-18 (CPU, GPU)
 	- Returns vector length 512
 - Alexnet (CPU, GPU)
 	- Returns vector length 4096

## Installation

Tested on Python 3.6

#### Dependencies

Pytorch: http://pytorch.org/

Pillow:  ```pip install Pillow```

##### For running the example, you will additionally need:
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
```python
from img_to_vec import Img2Vec
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)

# Read in an image
img = Image.open('test.jpg')
# Get a vector from img2vec
vec = img2vec.get_vec(img)
```
#### Img2Vec Params
**cuda** = (True, False) &nbsp; # Run on GPU? &nbsp; &nbsp; default: False<br>
**model** = ('resnet-18', 'alexnet') &nbsp; # Which model to use? &nbsp; &nbsp; default: 'resnet-18'<br>

## Advanced users 
----
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

## To-do
- Benchmark speed and accuracy
- Add ability to fine-tune on input data
- Export documentation to a normal place
- Package for Pip




