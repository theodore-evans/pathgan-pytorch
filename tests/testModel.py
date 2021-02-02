import torchtest
import unittest
import torch.functional as F

from modules.generative.Generator import Generator

class TestModel(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()
        self.loss_fn = F.mse_loss
    
    def test_suite(self):
        torchtest.test_suite(generator, )