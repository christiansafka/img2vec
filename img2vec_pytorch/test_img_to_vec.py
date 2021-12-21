import unittest
from PIL import Image
import numpy

from . import Img2Vec

class TestImg2Vec(unittest.TestCase):
    def test_default(self):
        img2vec = Img2Vec()
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(512, vec.size)

    def test_alexnet(self):
        img2vec = Img2Vec(model='alexnet')
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(4096, vec.size)

    def test_vgg(self):
        img2vec = Img2Vec(model='vgg')
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(4096, vec.size)

    def test_densenet(self):
        img2vec = Img2Vec(model='densenet')
        img = Image.open('./example/test_images/cat.jpg').convert('RGB')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1024, vec.size)

    def test_efficientnet_b0(self):
        img2vec = Img2Vec(model='efficientnet_b0')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1280, vec.size)

    def test_efficientnet_b1(self):
        img2vec = Img2Vec(model='efficientnet_b1')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1280, vec.size)

    def test_efficientnet_b2(self):
        img2vec = Img2Vec(model='efficientnet_b2')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1408, vec.size)

    def test_efficientnet_b3(self):
        img2vec = Img2Vec(model='efficientnet_b3')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1536, vec.size)

    def test_efficientnet_b4(self):
        img2vec = Img2Vec(model='efficientnet_b4')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1792, vec.size)

    def test_efficientnet_b5(self):
        img2vec = Img2Vec(model='efficientnet_b5')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2048, vec.size)

    def test_efficientnet_b6(self):
        img2vec = Img2Vec(model='efficientnet_b6')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2304, vec.size)

    def test_efficientnet_b7(self):
        img2vec = Img2Vec(model='efficientnet_b7')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(2560, vec.size)

if __name__ == "__main__":
    unittest.main()