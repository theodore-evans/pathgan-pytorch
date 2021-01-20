import unittest
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from models.generative.Discriminator import DiscriminatorResnet

class TestConvolutional(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.rand((64, 3, 224, 224))
        self.disc = DiscriminatorResnet()
        self.epsilon = 1e-5

    # Self fulfilling prohecy, makes sure orthogonal init and orthogonal reg_loss are working
    def test_ortho_reg(self):
        reg_loss_without_init = self.disc.get_orthogonal_reg_loss()
        for name, param in self.disc.named_parameters():
                if 'bias' not in name and 'Scale' not in name and 'gamma' not in name or 'Scale' in name and 'kernel' in name:
                    nn.init.orthogonal(param)
        reg_loss = self.disc.get_orthogonal_reg_loss()

        assert reg_loss_without_init > reg_loss, "Unitialized weights should have higher loss"
        assert reg_loss < self.epsilon , "Orthogonal Reg Loss should be close to 0"

        self.disc.ortho_reg_scale = 1e-2
        reg_loss_new = self.disc.get_orthogonal_reg_loss()
        assert reg_loss_new - reg_loss * 100 < self.epsilon, "New loss should be 100 times the old loss"