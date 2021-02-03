from modules.normalization.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from modules.blocks.ResidualBlock import ResidualBlock
from modules.blocks.DenseBlock import DenseBlock
from modules.blocks.ConvolutionalBlock import ConvolutionalBlock, DownscaleBlock, UpscaleBlock
from modules.blocks.AttentionBlock import AttentionBlock

import unittest
from modules.generative.Discriminator import DiscriminatorResnet
from collections import OrderedDict
import torch.nn as nn


class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.pathgan_discriminator = DiscriminatorResnet()
        self.discriminator_blocks_conv = self.pathgan_discriminator.conv_part.named_children()
        self.discriminator_blocks_dense = self.pathgan_discriminator.dense_part.named_children()

        self.parts = ['conv_part', 'dense_part']

        self.pathgan_blocks_conv = OrderedDict({
            ("residual_block_0", ResidualBlock): 1,
            ("downscale_block_0", DownscaleBlock): 2,
            ("residual_block_1", ResidualBlock): 3,
            ("downscale_block_1", DownscaleBlock): 4,
            ("residual_block_2", ResidualBlock): 5,
            ("downscale_block_2", DownscaleBlock): 6,
            ("residual_block_3", ResidualBlock): 7,
            ("attention_block", AttentionBlock): 8,
            ("downscale_block_3", DownscaleBlock): 9,
            ("residual_block_4", ResidualBlock): 10,
            ("downscale_block_4", DownscaleBlock): 11,
        })

        self.pathgan_blocks_dense = OrderedDict({
            ("dense_block_0", DenseBlock): 12,
            ("dense_block_1", DenseBlock): 13
        })

        self.pathgan_channels_conv = dict({
            "residual_block_0": (3, 3),
            "downscale_block_0": (3, 32),
            "residual_block_1": (32, 32),
            "downscale_block_1": (32, 64),
            "residual_block_2": (64, 64),
            "downscale_block_2": (64, 128),
            "residual_block_3": (128, 128),
            "attention_block": (128, 128),
            "downscale_block_3": (128, 256),
            "residual_block_4": (256, 256),
            "downscale_block_4": (256, 512)
        })

        self.pathgan_channels_dense = dict({
            "dense_block_0": (7*7*512, 512),
            "dense_block_1": (512, 1)
        })

    def test_that_number_of_blocks_is_correct(self):
        self.assertEqual(sum(1 for _ in self.discriminator_blocks_conv), len(self.pathgan_blocks_conv),
                         "Number of blocks should match configuration")

        self.assertEqual(sum(1 for _ in self.discriminator_blocks_dense), len(self.pathgan_blocks_dense),
                         "Number of blocks should match configuration")

    def test_that_block_order_is_correct_conv(self):
        for disc_block, correct_block in zip(self.discriminator_blocks_conv, self.pathgan_blocks_conv):
            disc_block_type = type(disc_block[1])
            correct_block_type = correct_block[1]
            self.assertIs(disc_block_type, correct_block_type,
                          f"Type and order of generator blocks should match configuration ({disc_block[0]})")

    def test_that_block_order_is_correct_dense(self):
        for disc_block, correct_block in zip(self.discriminator_blocks_dense, self.pathgan_blocks_dense):
            disc_block_type = type(disc_block[1])
            correct_block_type = correct_block[1]
            self.assertIs(disc_block_type, correct_block_type,
                          f"Type and order of generator blocks should match configuration ({disc_block[0]})")

    def test_that_block_names_are_correct(self):
        for disc_block, correct_block in zip(self.discriminator_blocks_conv, self.pathgan_blocks_conv):
            disc_block_name = disc_block[0]
            correct_block_name = correct_block[0]
            self.assertEqual(disc_block_name, correct_block_name,
                             f"Name of generator blocks should match configuration ({disc_block_name})")

        for disc_block, correct_block in zip(self.discriminator_blocks_dense, self.pathgan_blocks_dense):
            disc_block_name = disc_block[0]
            correct_block_name = correct_block[0]
            self.assertEqual(disc_block_name, correct_block_name,
                             f"Name of generator blocks should match configuration ({disc_block_name})")

    def test_that_blocks_have_correct_activations(self):
        for block in self.pathgan_discriminator.named_children():
            if type(block[1]) in (DenseBlock, ConvolutionalBlock) and block[0] != 'sigmoid_block':
                self.assertIs(type(block[1].activation), nn.LeakyReLU)

    def test_that_blocks_have_correct_in_and_out_channels(self):
        for name, block in self.discriminator_blocks_conv:
            block_channels = (block.in_channels, block.out_channels)
            self.assertEqual(block_channels, self.pathgan_channels_conv[name],
                             f"Blocks should have correct in_ and out_channels")

        for name, block in self.discriminator_blocks_dense:
            block_channels = (block.in_channels, block.out_channels)
            self.assertEqual(block_channels, self.pathgan_channels_dense[name],
                             f"Blocks should have correct in_ and out_channels")
