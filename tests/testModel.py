
from modules.initialization.XavierInitialization import XavierInitialization
from tests.torchtest import assert_vars_change, test_suite

import unittest
import math
import torch
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.init import _calculate_fan_in_and_fan_out

from modules.generative.Generator import Generator
from modules.utils import get_parameters

class TestModel(unittest.TestCase):
    def setUp(self):
        self.generator = Generator(initialization=XavierInitialization)
        self.loss_fn = F.mse_loss
        inputs = Variable(torch.rand(1, 200))
        latent_input = Variable(torch.rand(1,200))
        target = Variable(torch.rand(1, 3, 224, 224))
        self.batch = [inputs, latent_input, target]
        self.device = torch.device('cpu')
    
    def test_weights_are_xavier_initialized(self):
        model = self.generator
        weights = [w[1] for w in get_parameters(model, 'weight', recurse = True)]
        for weight in weights:
            fan_in, fan_out = _calculate_fan_in_and_fan_out(weight)
            max_std = math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * max_std
            self.assertTrue(torch.all(torch.le(weight, a)))
            self.assertTrue(torch.all(torch.ge(weight, -a)))
            
        biases = [b[1] for b in get_parameters(model, 'bias', recurse = True)]
        for bias in biases:
            self.assertTrue(torch.all(torch.eq(bias, 0)))
    
    def test_weights_change_due_to_training(self):
        model = self.generator
        weights = get_parameters(model, 'weight', recurse = True)
        assert_vars_change(model,
                           self.loss_fn,
                           optim = torch.optim.Adam(model.parameters()),
                           batch = self.batch,
                           device = self.device,
                           params = weights)
    
    def test_biases_change_due_to_training(self):
        model = self.generator
        biases = get_parameters(model, 'bias', recurse = True)
        assert_vars_change(model,
                           self.loss_fn,
                           optim = torch.optim.Adam(model.parameters()),
                           batch = self.batch,
                           device = self.device,
                           params = biases)
        
    def test_for_NaN_values(self):
        model = self.generator
        test_suite(model,
                   self.loss_fn,
                   optim=torch.optim.Adam(model.parameters()),
                   batch=self.batch,
                   test_nan_vals=True)
        
    def test_for_Inf_values(self):
        model = self.generator
        test_suite(model,
                   self.loss_fn,
                   optim=torch.optim.Adam(model.parameters()),
                   batch=self.batch,
                   test_inf_vals=True)