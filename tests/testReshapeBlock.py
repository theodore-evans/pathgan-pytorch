import unittest
import torch
from modules.blocks.ReshapeBlock import ReshapeBlock

class TestReshapeBlock(unittest.TestCase):
    def setUp(self):
        self.dense_output = torch.zeros(10, 12544)
        self.reshape = ReshapeBlock(12544, 256, 7)
    
    def test_checks_that_out_channels_divides_in_channels(self):
        with self.assertRaises(ValueError):
            ReshapeBlock(12544, 257, (7,7))
        
    def test_checks_that_in_channels_factorises_into_out_channels_and_image_shape(self):
        with self.assertRaises(ValueError):
            ReshapeBlock(12544, 256, (6,6))
            
    def test_checks_that_in_channels_matches_input_shape(self):
        reshape = ReshapeBlock(9216, 256, (7,7))
        with self.assertRaises(ValueError):
            reshape(self.dense_output)
            
    def test_output_is_reshaped(self):
        output = self.reshape(self.dense_output)
        self.assertEqual(output.shape, (self.dense_output.size(0), 256, 7, 7))
