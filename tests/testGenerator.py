from modules.blocks.ReshapeBlock import ReshapeBlock
from modules.normalization.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from modules.blocks.ResidualBlock import ResidualBlock
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock, UpscaleBlock
from modules.blocks.AttentionBlock import AttentionBlock

import unittest
from modules.generative.Generator import Generator
import torch.nn as nn

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.pathgan_generator = Generator()
        self.generator_blocks = self.pathgan_generator.named_children()

        self.pathgan_blocks = [
            ("dense_block_0", DenseBlock),
            ("dense_block_1", DenseBlock),
            ("reshape_block", ReshapeBlock),
            ("res_block_0", ResidualBlock),
            ("upscale_block_0", UpscaleBlock),
            ("res_block_1", ResidualBlock),
            ("upscale_block_1", UpscaleBlock), 
            ("res_block_2", ResidualBlock),
            ("attention_block_2", AttentionBlock),
            ("upscale_block_2", UpscaleBlock), 
            ("res_block_3", ResidualBlock),
            ("upscale_block_3", UpscaleBlock),
            ("res_block_4", ResidualBlock),
            ("upscale_block_4", UpscaleBlock),  
            ("sigmoid_block", ConvolutionalBlock)
        ]
        
        self.pathgan_channels = dict({
            "dense_block_0" : (200,1024),
            "dense_block_1" : (1024,12544),
            "reshape_block" : (12544, 256),

            "res_block_0" : (256,256),
            "upscale_block_0" : (256,512),
            
            "res_block_1" : (512,512),
            "upscale_block_1" : (512,256),
            
            "res_block_2" : (256,256),
            "attention_block_2" : (256,256),
            "upscale_block_2" : (256,128),
            
            "res_block_3" : (128,128),
            "upscale_block_3" : (128,64),
            
            "res_block_4" : (64,64),
            "upscale_block_4" : (64,32),
            
            "sigmoid_block" : (32,3),
        })
    
    def test_that_number_of_blocks_is_correct(self):
        self.assertEqual(sum(1 for _ in self.generator_blocks), len(self.pathgan_blocks),
                         "Number of blocks should match configuration")
    
    def test_that_block_order_is_correct(self):
        for generator_block, correct_block in zip(self.generator_blocks, self.pathgan_blocks):
            generator_block_type = type(generator_block[1])
            correct_block_type = correct_block[1]
            self.assertIs(generator_block_type, correct_block_type,
                          f"Type and order of generator blocks should match configuration ({generator_block[0]})")
            
    def test_that_block_names_are_correct(self):
        for generator_block, correct_block in zip(self.generator_blocks, self.pathgan_blocks):
            generator_block_name = generator_block[0]
            correct_block_name = correct_block[0]
            self.assertEqual(generator_block_name, correct_block_name, 
                          f"Name of generator blocks should match configuration ({generator_block_name})")
            
    def test_that_blocks_have_correct_activations(self):
        for block in self.pathgan_generator.named_children():
            if type(block[1]) in (DenseBlock, ConvolutionalBlock) and block[0] != 'sigmoid_block':
                self.assertIs(type(block[1].activation), nn.LeakyReLU)
                
    def test_that_blocks_have_adaptive_instance_normalization(self):
        for block in self.pathgan_generator.children():
            if type(block) in (DenseBlock, UpscaleBlock):
                self.assertTrue(hasattr(block, 'normalization'),
                                f"Dense and convolution blocks should have normalization ({block})")
                self.assertIs(type(block.normalization), AdaptiveInstanceNormalization,
                              f"Normalization should be AdaIN ({block})")
                
            if type(block) is ResidualBlock:
                for res_block in block.children():
                    self.assertIs(type(res_block.normalization), AdaptiveInstanceNormalization, 
                                  f"Normalization should be AdaIN ({res_block})")
                
    def test_that_blocks_have_correct_in_and_out_channels(self):
        for name, block in self.pathgan_generator.named_children():
            block_channels = (block.in_channels, block.out_channels)
            self.assertEqual(block_channels, self.pathgan_channels[name],
                             "Blocks should have correct in_ and out_channels")
            
    def test_checks_for_valid_output_image_size(self):
        with self.assertRaises(ValueError):
            Generator(synthesis_out_channels = [512, 256, 128, 64, 32], output_image_size = 223)