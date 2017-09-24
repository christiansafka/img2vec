import os
import argparse

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

class Img2Vec():

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """
            Parameters:
                see docs: https://github.com/christiansafka/img2vec
        """
        self.cuda = cuda
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        if self.cuda:
            self.model.cuda()

        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img):
        """
            Parameters:
                img: PIL image
            Returns:
                Numpy vector representing input image
        """
        if self.cuda:
            image = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)).cuda()
        else:
            image = Variable(self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0))

        my_embedding = torch.zeros(self.layer_output_size)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        return my_embedding.numpy()

    def _get_model_and_layer(self, model_name, layer):
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)

# Arguments for using img2vec as a CLI
parser = argparse.ArgumentParser(description='Image to vector embedding with PyTorch')

parser.add_argument('--input', '-i', metavar='PATH', required=True, help='path to directory of images')
parser.add_argument('--output', '-o', metavar='OUTPUT', default='output.csv',
                    help='csv to store image filenames and vector representations - Default: output.csv')
parser.add_argument('--model', metavar='MODEL', default='resnet-18', help='model architecture - Default: resnet-18')
parser.add_argument('--layer', metavar='LAYER', default='avgpool', help='Select a different layer from where to pull embeddings')
parser.add_argument('--layer_output_size', metavar='VECSIZE', default=512, help='Provide the correct output size for a selected layer')
parser.add_argument('--cuda', '-gpu', metavar='CUDA', default=False, help='Run the model on GPU - Default: False')

args = parser.parse_args()

def main():
    img2vec = Img2Vec(cuda=args.cuda, model=args.model, layer=args.layer, layer_output_size=args.layer_output_size)
    
    with open(args.output, 'w') as f:
        for file in tqdm(os.listdir(args.input)):
            filename = os.fsdecode(file)
            image = Image.open(os.path.join(args.input, filename))

            row = filename + ',' + str(img2vec.get_vec(image)) + '\n'
            f.write(row)


if __name__ == "__main__":
    from PIL import Image
    from tqdm import tqdm
    main()