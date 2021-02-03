import unittest
import torch
from torch import nn

from modules.generative.Discriminator import DiscriminatorResnet
from modules.utils import apply_same_padding

class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.data = torch.rand((64, 3, 224, 224))
        self.epsilon = 1e-5

    # Self fulfilling prohecy, makes sure orthogonal init and orthogonal reg_loss are working
    def test_ortho_reg(self):
        disc = DiscriminatorResnet()
        reg_loss_without_init = disc.get_orthogonal_reg_loss()
        for name, param in disc.named_parameters():
                if 'bias' not in name and 'scale' not in name and 'gamma' not in name or 'Scale' in name and 'kernel' in name:
                    nn.init.orthogonal(param)
        reg_loss = disc.get_orthogonal_reg_loss()

        assert reg_loss_without_init > reg_loss, "Unitialized weights should have higher loss"
        assert reg_loss < self.epsilon , "Orthogonal Reg Loss should be close to 0"

        disc.ortho_reg_scale = 1e-2
        reg_loss_new = disc.get_orthogonal_reg_loss()
        assert reg_loss_new - reg_loss * 100 < self.epsilon, "New loss should be 100 times the old loss"
        
    def test_same_padding(self):
        conv_layer = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(3,3), stride=1)
        output = conv_layer(self.data)
        assert output.size(2) != self.data.size(2), "Output height should not match input height"
        assert output.size(3) != self.data.size(3), "Output width should not match input width"
        
        apply_same_padding(conv_layer)
        same_padding_output = conv_layer(self.data)
        
        assert same_padding_output.size(2) == self.data.size(2), "Output height should match input height"
        assert same_padding_output.size(3) == self.data.size(3), "Output width should match input width"