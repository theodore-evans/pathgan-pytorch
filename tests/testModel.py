from torchtest import assert_vars_change
import unittest
import torch
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

from modules.generative.Generator import Generator

class TestModel(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()
        self.loss_fn = F.mse_loss
        inputs = Variable(torch.rand(1, 200))
        latent_input = Variable(torch.rand(1,200))
        target = Variable(torch.rand(1, 3, 224, 224))
        self.batch = [inputs, latent_input, target]
        self.device = torch.device('cpu')
    
    def test_weights_change_due_to_training(self):
        model = self.generator
        weights = [ np[1] for np in model.named_parameters() if np[1] is 'weight' ]
        assert_vars_change(model,
                           self.loss_fn,
                           optim = torch.optim.Adam(model.parameters()),
                           batch = self.batch,
                           device = self.device,
                           params = weights)
        