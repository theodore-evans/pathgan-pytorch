from typing import no_type_check

from torch.nn.modules import loss
from .Mapping import Mapping
from .Discriminator import DiscriminatorResnet
from .Generator import Generator
import torch
from torch import optim, nn
from dataset import Dataset
from itertools import chain
from modules.regularization.OrthogonalRegularizer import OrthogonalRegularizer

# Should we extend nn.Module here?
# I don't think it's necessary now


class PathologyGAN(nn.Module):
    def __init__(self, dataset: Dataset, learning_rate_d: float, learning_rate_g: float, beta: float, epochs: int, z_dim: int, checkpoint_path: str):
        super().__init__()
        self.dataset = dataset
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.initialize_optimizers(learning_rate_d, learning_rate_g, beta)
        self.regularizer = OrthogonalRegularizer(1e-4)
        
        self.checkpoint_path = checkpoint_path
        self.epochs = epochs
        self.z_dim = z_dim

        self.multi_label_margin_loss = nn.MultiLabelSoftMarginLoss()

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
        self.load_state_dict(self.checkpoint_path)

    def store_weights(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_path)

    # Output images after each epoch with sampling from z
    def generate_sample_images(self) -> torch.Tensor:
        pass

    def generate_images(self, input: torch.Tensor) -> torch.Tensor:
        pass

    # May also return a tuple of tensors
    def calculate_losses(self) -> torch.Tensor:
        pass

    def train_discriminator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:

        self.disc.zero_grad()
        out_real = self.disc(real_images)
        out_fake = self.disc(fake_images.detach())

        real_fake_diff = out_real - torch.mean(out_fake, dim=0, keepdim=True)
        fake_real_diff = out_fake - torch.mean(out_real, dim=0, keepdim=True)

        loss_dis_real = self.multi_label_margin_loss(
            real_fake_diff, torch.ones(*real_fake_diff.shape))
        loss_dis_fake = self.multi_label_margin_loss(
            fake_real_diff, torch.zeros(*real_fake_diff.shape))

        # TODO: Should we implement gradient penalty?
        orthogonality_loss = self.regularizer.get_regularizer_loss(self.disc)
        loss_discriminator = loss_dis_fake + loss_dis_real + orthogonality_loss
        loss_discriminator.backward()

        self.optimizerD.step()

        return loss_discriminator.item()

    def train_generator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        self.gen.zero_grad()
        self.mapping.zero_grad()
        out_real = self.disc(real_images)
        out_fake = self.disc(fake_images)

        real_fake_diff = out_real - torch.mean(out_fake, dim=0, keepdim=True)
        fake_real_diff = out_fake - torch.mean(out_real, dim=0, keepdim=True)

        loss_gen_real = self.multi_label_margin_loss(
            fake_real_diff, torch.ones(*real_fake_diff.shape))
        loss_gen_fake = self.multi_label_margin_loss(
            real_fake_diff, torch.zeros(*real_fake_diff.shape))

        orthogonality_loss = self.regularizer.get_regularizer_loss(self.gen) + self.regularizer.get_regularizer_loss(self.mapping)

        loss_generator = loss_gen_fake + loss_gen_real + orthogonality_loss
        loss_generator.backward()

        self.optimizerG.step()

        return loss_generator.item()

    def train(self, restore: bool = False, n_critic: int = 5) -> None:
        iters = 0

        # Create a steady latent input here

        if restore:
            self.load_weights()

        print("Starting Training Loop")

        for epoch in range(self.epochs):
            for images, labels in self.dataset:
                
                z_latent = torch.randn(
                    len(images), self.z_dim, device=self.device)

                batch_images = images.to(self.device)

                w_latent = self.mapping(z_latent)
                fake_images = self.gen(w_latent,w_latent)

                self.train_discriminator(batch_images, fake_images)

                if iters % n_critic == 0:
                    self.train_generator(batch_images, fake_images)
                iters += 1

            self.store_weights()
            self.dataset.reset()
            self.generate_sample_images()

        print("Training Finished")
