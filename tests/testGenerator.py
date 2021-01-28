from models.generative.ResidualBlock import ResidualBlock
from models.generative.DenseBlock import DenseBlock
from models.generative.ConvolutionalBlock import ConvolutionalBlock
from models.generative.AttentionBlock import AttentionBlock

import unittest
from models.generative.Generator import Generator
from collections import OrderedDict

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.pathgan_generator = Generator()
        self.generator_blocks = self.pathgan_generator.named_children()

        self.pathgan_blocks = OrderedDict({
            ("dense_block_1", DenseBlock) : 1,
            ("dense_block_2", DenseBlock) : 2,
            
            ("res_block_1", ResidualBlock) : 3,
            ("upscale_block_1", ConvolutionalBlock) : 4,
            
            ("res_block_2", ResidualBlock) : 5,
            ("upscale_block_2", ConvolutionalBlock) : 6,
            
            ("res_block_3", ResidualBlock) : 7,
            ("attention_block_3", AttentionBlock) : 8,
            ("upscale_block_3", ConvolutionalBlock) : 9,
            
            ("res_block_4", ResidualBlock) : 10,
            ("upscale_block_4", ConvolutionalBlock) : 11,
            
            ("res_block_5", ResidualBlock) : 12,
            ("upscale_block_5", ConvolutionalBlock) : 13,
            
            ("sigmoid_block", ConvolutionalBlock) : 14
        })
    
    def test_that_number_of_blocks_is_correct(self):
        self.assertEqual(sum(1 for _ in self.generator_blocks), len(self.pathgan_blocks),
                         "Number of blocks should match configuration")
    
    def test_that_block_order_is_correct(self):      
        for generator_block, correct_block in zip(self.generator_blocks, self.pathgan_blocks):
            generator_block_type = type(generator_block[1])
            correct_block_type = correct_block[1]
            self.assertIs(generator_block_type, correct_block_type, 
                          "Type and order of generator blocks should match configuration")
            
    # def test_that_block_names_are_correct(self):        
    #     for generator_block, correct_block in zip(self.generator_blocks, self.pathgan_blocks):
    #         generator_block_name = generator_block[0]
    #         correct_block_name = correct_block[0]
    #         self.assertIs(generator_block_name, correct_block_name, 
    #                       "Name of generator blocks should match configuration")