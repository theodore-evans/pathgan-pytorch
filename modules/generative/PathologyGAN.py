from typing import no_type_check
from .Mapping import Mapping
from .Discriminator import DiscriminatorResnet
from .Generator import Generator
import torch
from torch import optim
from dataset import Dataset
from itertools import chain
from modules.regularization import OrthogonalRegularizer

# Should we extend nn.Module here?
# I don't think it's necessary now


class PathologyGAN:
    def __init__(self, dataset: Dataset, learning_rate_d: float, learning_rate_g: float, beta: float, epochs: int, z_dim: int):
        self.dataset = dataset
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.initialize_optimizers(learning_rate_d, learning_rate_g, beta)
        self.regularizer = OrthogonalRegularizer(1e-4)
        self.epochs = epochs
        self.z_dim = z_dim

    # Initialize the optimizer here?

    def build_model(self):
        self.mapping = Mapping()
        self.disc = DiscriminatorResnet()
        self.gen = Generator()

    def initialize_optimizers(self, learning_rate_d: float, learning_rate_g: float, beta: float):
        disc_parameters = self.disc.parameters()
        gen_parameters = chain(self.mapping.parameters(),
                               self.gen.parameters())
        self.optimizerD = optim.Adam(
            disc_parameters, lr=learning_rate_d, betas=(beta, 0.999))
        self.optimizerG = optim.Adam(
            gen_parameters, lr=learning_rate_g, betas=(beta, 0.999))

    def load_weights(self) -> None:
        pass

    def store_weights(self) -> None:
        pass

    # Output images after each epoch with sampling from z
    def generate_sample_images(self) -> torch.Tensor:
        pass

    def generate_images(self, input: torch.Tensor) -> torch.Tensor:
        pass

    # May also return a tuple of tensors
    def calculate_losses(self) -> torch.Tensor:
        pass


    def train(self, restore: bool = False, n_critic: int = 5) -> None:
        iters = 0
        # Restore model here
        # Steady latent input ?

        if restore:
            self.load_weights()

        for epoch in range(self.epochs):
            for images, labels in self.dataset:
                pass

                # Update discriminator
                z_latent = torch.randn(len(images), self.z_dim, device=self.device)

                # TODO: implement losses and split them into functions
                # Update generator every n iters
                if iters % n_critic == 0:
                    pass
                iters += 1

            self.store_weights()
            self.dataset.reset()
            self.generate_sample_images()
