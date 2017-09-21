import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

class Img2Vec():
    def __init__(self, cuda=False):
        self.cuda = cuda

        if self.cuda:
            self.resnet_18 = models.resnet18(pretrained=True).cuda()
        else:
            self.resnet_18 = models.resnet18(pretrained=True)

        self.resnet_18.eval()

        self.avgpool_layer = self.resnet_18._modules.get('avgpool')
        self.scaler = transforms.Scale((224, 224))
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img):
        if self.cuda:
            image = Variable(self.to_tensor(self.scaler(img)).unsqueeze(0)).cuda()
        else:
            image = Variable(self.to_tensor(self.scaler(img)).unsqueeze(0))

        my_embedding = torch.zeros(512)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.avgpool_layer.register_forward_hook(copy_data)
        h_x = self.resnet_18(image)
        h.remove()

        return my_embedding.numpy()
