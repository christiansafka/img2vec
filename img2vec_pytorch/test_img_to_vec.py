import unittest
from PIL import Image
import numpy


from . import Img2Vec

class TestImg2Vec(unittest.TestCase):
    def test_default(self):
        img2vec = Img2Vec()
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(512, vec.size)

    def test_alexnet(self):
        img2vec = Img2Vec(model='alexnet')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(4096, vec.size)

    def test_vgg(self):
        img2vec = Img2Vec(model='vgg')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(4096, vec.size)

    def test_densenet(self):
        img2vec = Img2Vec(model='densenet')
        img = Image.open('./example/test_images/cat.jpg')
        vec = img2vec.get_vec(img)
        self.assertEqual(True, isinstance(vec, numpy.ndarray))
        self.assertEqual(1, vec.ndim)
        self.assertEqual(1024, vec.size)

if __name__ == "__main__":
    unittest.main()