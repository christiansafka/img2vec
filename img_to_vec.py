import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import time

torch.utils.backcompat.broadcast_warning.enabled = False
torch.utils.backcompat.keepdim_warning.enabled = False

class Img2Vec():
    def __init__(self, cuda=False):
        self.cuda = cuda

        if self.cuda:
            self.resnet_18 = models.resnet18(pretrained=True).cuda()
        else:
            self.resnet_18 = models.resnet18(pretrained=True)

        # for param in self.resnet_18.parameters():
        #     param.requires_grad = False
        self.resnet_18.eval()

        self.avgpool_layer = self.resnet_18._modules.get('avgpool')
        self.scaler = transforms.Scale((224, 224))
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img):
        if self.cuda:
            image = Variable(self.to_tensor(self.scaler(img)).float().div(255).unsqueeze(0)).cuda()
        else:
            image = Variable(self.to_tensor(self.scaler(img)).float().div(255).unsqueeze(0))

        my_embedding = torch.zeros(512)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.avgpool_layer.register_forward_hook(copy_data)
        h_x = self.resnet_18(image)
        h.remove()

        return my_embedding.numpy()

# a = Img2Vec()
# scaler = transforms.Scale((224, 224))
# to_tensor = transforms.ToTensor()
# image = Variable(to_tensor(scaler(Image.open("test.jpg"))).float().div(255).unsqueeze(0))
# print('ready')
# t0 = time.time()
# print(a.get_vec(image))
# t1 = time.time()
# print('took %d' % (t1-t0))
#print(a.get_vec(image))
