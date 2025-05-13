import torch
from torchvision.transforms import v2
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import vgg13_bn, VGG13_BN_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import densenet161, DenseNet161_Weights
from torchvision.models import densenet169, DenseNet169_Weights
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


import numpy as np

class Img2Vec():
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048
    }

    EFFICIENTNET_OUTPUT_SIZES = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1408,
        'efficientnet_b3': 1536,
        'efficientnet_b4': 1792,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2304,
        'efficientnet_b7': 2560
    }
    
    DENSENET_OUTPUT_SIZE = {
        'densenet121' : 1024,
        'densenet161' : 2208,
        'densenet169' : 1664,
        'densenet201' : 1920
    }

    def __init__(self, cuda=False, model='resnet', layer='default', layer_output_size=512, gpu=0):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float,scale=True),
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image or list of PIL Images
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.transforms(im) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name in ['alexnet', 'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == 'densenet' or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                if self.model_name in ['alexnet', 'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                    # For VGG models, reshape the output tensor to 2D before copying
                    if len(o.data.shape) == 4:
                        my_embedding.copy_(o.data.squeeze(-1).squeeze(-1))
                    else:
                        my_embedding.copy_(o.data)
                else:
                    my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                    return my_embedding.numpy()
                elif self.model_name.startswith('densenet') or 'efficientnet' in self.model_name:
                    return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
                else:
                    return my_embedding.numpy()[:, :, 0, 0]
        else:
            try:
                # Validate and transform the image
                image = self.transforms(img).unsqueeze(0).to(self.device)
            except Exception as e:
                raise ValueError(f"Invalid image input: {e}")

            if self.model_name in ['alexnet', 'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                my_embedding = torch.zeros(1, self.layer_output_size)
            elif self.model_name.startswith('densenet') or 'efficientnet' in self.model_name:
                my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                if self.model_name in ['alexnet', 'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                    # For VGG models, reshape the output tensor to 2D before copying
                    if len(o.data.shape) == 4:
                        my_embedding.copy_(o.data.squeeze(-1).squeeze(-1))
                    else:
                        my_embedding.copy_(o.data)
                else:
                    my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(image)
            h.remove()

            if tensor:
                return my_embedding
            else:
                if self.model_name in ['alexnet', 'vgg', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
                    return my_embedding.numpy()[0, :]
                elif self.model_name.startswith('densenet'):
                    return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
                else:
                    return my_embedding.numpy()[0, :, 0, 0]

    def _get_resnet_model(self, model_name):
        """ Helper function to get ResNet model based on model_name """
        if model_name == 'resnet18':
            return resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet34':
            return resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_name == 'resnet50':
            return resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            return resnet101(weights=ResNet101_Weights.DEFAULT)
        elif model_name == 'resnet152':
            return resnet152(weights=ResNet152_Weights.DEFAULT)
        
    def _get_vgg_model(self, model_name):
        """ Helper function to get VGG model based on model_name """
        if model_name == 'vgg11':
            return vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
        elif model_name == 'vgg13':
            return vgg13_bn(weights=VGG13_BN_Weights.DEFAULT)
        elif model_name == 'vgg16':
            return vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        elif model_name == 'vgg19':
            return vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            
    def _get_densenet_model(self, model_name):
        """ Helper function to get DenseNet model based on model_name """
        if model_name == 'densenet121':
            return densenet121(weights=DenseNet121_Weights.DEFAULT)   
        elif model_name == 'densenet161':
            return densenet161(weights=DenseNet161_Weights.DEFAULT)
        elif model_name == 'densenet169':
            return densenet169(weights=DenseNet169_Weights.DEFAULT)  
        elif model_name == 'densenet201':
            return densenet201(weights=DenseNet201_Weights.DEFAULT)     

    def _get_efficientnet_model(self, model_name):
        """ Helper function to get EfficientNet model based on model_name """
        if model_name == 'efficientnet_b0':
            return efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        elif model_name == 'efficientnet_b1':
            return efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        elif model_name == 'efficientnet_b2':
            return efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        elif model_name == 'efficientnet_b3':
            return efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        elif model_name == 'efficientnet_b4':
            return efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        elif model_name == 'efficientnet_b5':
            return efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        elif model_name == 'efficientnet_b6':
            return efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
        elif model_name == 'efficientnet_b7':
            return efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

    def _get_model_and_layer(self, model_name, layer):
    
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """

        if model_name == 'resnet':
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES['resnet18']
            else:
                layer = model._modules.get(layer)
            return model, layer
        
        elif model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            model = self._get_resnet_model(model_name)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = alexnet(weights=AlexNet_Weights.DEFAULT)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer
        
        elif model_name == 'vgg':
            model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features # should be 4096
            else:
                layer = model.classifier[-layer]
            return model, layer
                
        elif model_name in ['vgg11','vgg13', 'vgg16', 'vgg19']:
            model = self._get_vgg_model(model_name)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features
            else:
                layer = model.classifier[-layer]
            return model, layer

        elif model_name == 'densenet':
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer
        
        elif model_name in ['densenet121', 'densenet161', 'densenet169', 'densenet201']:
            model = self._get_densenet_model(model_name)
            
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = self.DENSENET_OUTPUT_SIZE[model_name]
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer

        elif "efficientnet" in model_name:
            
            if model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']:
                model = self._get_efficientnet_model(model_name)
            else:
                raise KeyError('Un support %s.' % model_name)

            if layer == 'default':
                layer = model.features
                self.layer_output_size = self.EFFICIENTNET_OUTPUT_SIZES[model_name]
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)
